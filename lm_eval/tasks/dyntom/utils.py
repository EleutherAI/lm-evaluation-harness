import functools
import json
import re

import datasets
from huggingface_hub import hf_hub_download

SUBJECTS = ("belief", "emotion", "intention", "action")
_SUBJ_RE = re.compile(r"\bthe (belief|emotion|intention|action) of\b", re.I)

# ---------- LOADER (cache-once; one doc per 5-scenario trial) ----------

@functools.lru_cache(maxsize=1)
def _build_docs():
    path = hf_hub_download("YangXiao-nlp/DynToM", "DynToM.json", repo_type="dataset")
    data = json.load(open(path, encoding="utf-8"))
    docs = []
    for tid, trial in data.items():
        stage = trial["stage"]
        if stage.get("scenario numbers") != 5:
            continue
        questions = trial["question"]
        meta = []
        prompt_qs = {}
        for qid, q in questions.items():
            family = re.sub(r"_\d+$", "", qid)
            dim, subject = _classify(family, q["question"])
            meta.append({
                "id": qid,
                "gold": q["true answer"].strip().lower(),
                "family": family,
                "dim": dim,
                "subject": subject,
            })
            prompt_qs[qid] = {"question": q["question"], "options": q["options"]}
        docs.append({
            "trial_id": tid,
            "characters_information": stage["characters information"],
            "story": json.dumps(stage["story"], ensure_ascii=False, indent=2),
            "questions_new": json.dumps(prompt_qs, ensure_ascii=False, indent=2),
            # meta stored as JSON string to avoid Arrow nested-schema issues (subject is sometimes None)
            "meta": json.dumps(meta),
            # gold dict for logging only — never used in scoring
            "target": json.dumps({m["id"]: m["gold"] for m in meta}),
        })
    return docs


def load(**kwargs):
    return {"train": datasets.Dataset.from_list(_build_docs())}


def _classify(family, text):
    if family == "type_c_how":
        return "influence", None
    m = _SUBJ_RE.search(text)
    if not m:
        raise ValueError(f"no subject parsed from: {text!r}")
    subj = m.group(1).lower()
    return ("u" if family == "type_a_what" else "t"), subj


# ---------- PROMPT FUNCTIONS (verbatim Fig 17; shared tail) ----------

_VANILLA = "Answer the questions based on the story. "
_COT = (
    "Answer the questions based on the story; first, think step by step, analyze the answers "
    "to the questions, and finally, output the most likely answers. "
)
_TAIL = (
    ". Answer the question, and response in JSON format:"
    '{[question_id]:[a, b, c or d]}. for example: {"type_d_how_1":"a"}'
)


def _render(doc, head):
    return (
        f'{head}{doc["characters_information"]} \n'
        f'{doc["story"]} \n'
        f'{doc["questions_new"]}{_TAIL}'
    )


def doc_to_text(doc):
    return _render(doc, _VANILLA)


def doc_to_text_cot(doc):
    return _render(doc, _COT)


# ---------- ANSWER EXTRACTION + SCORING ----------

# Matches bare or quoted key:value pairs — handles JSON and CoT prose
_PAIR_RE = re.compile(
    r'["\']?(type_[a-z_]+\d+)["\']?\s*:\s*["\']?([a-zA-Z])["\']?'
)
# Isolates the first {...} block (strips markdown fences and surrounding prose)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.S)


def _parse_answers(gen):
    out = {}
    # Try outermost JSON-like object first
    m = _JSON_OBJ_RE.search(gen)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                out = {k: str(v).strip().lower()[:1] for k, v in obj.items()}
        except Exception:
            pass
    # Regex fallback fills gaps (CoT models often emit prose alongside JSON)
    for qid, letter in _PAIR_RE.findall(gen):
        out.setdefault(qid, letter.strip().lower())
    return out


def process_results(doc, results):
    pred = _parse_answers(results[0])
    buckets = {}  # name -> [correct, total]

    def bump(name, ok):
        b = buckets.setdefault(name, [0, 0])
        b[0] += ok
        b[1] += 1

    for m in json.loads(doc["meta"]):
        ok = 1 if pred.get(m["id"]) == m["gold"] else 0
        bump("acc", ok)
        if m["dim"] == "influence":
            bump("acc_influence", ok)
        else:
            bump("acc_core", ok)
            bump(f"acc_{m['subject']}_{m['dim']}", ok)

    return {name: c / t for name, (c, t) in buckets.items()}


# ---------- PARTITION SUPPORT (per-bucket tasks) ----------
# Each partition task asks only one bucket's questions per trial (5-15 Q vs 71),
# so prompts are ~4-7x shorter. These are diagnostic — the model no longer reasons
# over all questions together. The faithful task is dyntom/dyntom_cot.

def _make_process_docs(subject, dim):
    """Return a process_docs fn that filters each trial doc to one question bucket."""
    def _process(dataset):
        def _transform(doc):
            full_meta = json.loads(doc["meta"])
            if dim == "influence":
                bucket_meta = [m for m in full_meta if m["dim"] == "influence"]
            else:
                bucket_meta = [m for m in full_meta
                               if m["subject"] == subject and m["dim"] == dim]
            full_qs = json.loads(doc["questions_new"])
            filtered_qs = {m["id"]: full_qs[m["id"]] for m in bucket_meta}
            return {
                "trial_id": doc["trial_id"],
                "characters_information": doc["characters_information"],
                "story": doc["story"],
                "questions_new": json.dumps(filtered_qs, ensure_ascii=False, indent=2),
                "meta": json.dumps(bucket_meta),
                "target": json.dumps({m["id"]: m["gold"] for m in bucket_meta}),
            }
        return dataset.map(_transform)
    return _process


def _make_process_results(metric_name):
    """Return a process_results fn that emits only the single named metric."""
    def _pr(doc, results):
        return {metric_name: process_results(doc, results).get(metric_name, 0.0)}
    return _pr


# Named functions for !function references in partition YAMLs
process_docs_belief_u    = _make_process_docs("belief",    "u")
process_docs_belief_t    = _make_process_docs("belief",    "t")
process_docs_emotion_u   = _make_process_docs("emotion",   "u")
process_docs_emotion_t   = _make_process_docs("emotion",   "t")
process_docs_intention_u = _make_process_docs("intention", "u")
process_docs_intention_t = _make_process_docs("intention", "t")
process_docs_action_u    = _make_process_docs("action",    "u")
process_docs_action_t    = _make_process_docs("action",    "t")
process_docs_influence   = _make_process_docs(None,        "influence")

process_results_belief_u    = _make_process_results("acc_belief_u")
process_results_belief_t    = _make_process_results("acc_belief_t")
process_results_emotion_u   = _make_process_results("acc_emotion_u")
process_results_emotion_t   = _make_process_results("acc_emotion_t")
process_results_intention_u = _make_process_results("acc_intention_u")
process_results_intention_t = _make_process_results("acc_intention_t")
process_results_action_u    = _make_process_results("acc_action_u")
process_results_action_t    = _make_process_results("acc_action_t")
process_results_influence   = _make_process_results("acc_influence")
