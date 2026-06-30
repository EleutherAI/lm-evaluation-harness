"""lm-eval adapter for FANToM (Kim et al., EMNLP 2023; arXiv:2310.15421).

Faithfulness over convenience: this file **vendors** FANToM's own loader, prompt
assembly, and scoring logic from the read-only submodule under
`benchmarks/fantom/` (never imported — submodules are not importable as packages).
Provenance of each vendored block is noted inline:

  - download/unzip      ← benchmarks/fantom/task/dataset_loader.py
  - prompt flatten       ← benchmarks/fantom/eval_fantom.py: setup_fantom / set_beliefQA_multiple_choices
  - per-QA scorers       ← benchmarks/fantom/eval_fantom.py: compute_f1, evaluate_*_q,
                            map_binary_answer_to_int, yesno_to_int, parse_response
  - corpus report        ← benchmarks/fantom/eval_fantom.py: run_reports / score_and_analyze

Architecture (see outputs/recon/fantom.md and the plan):
  doc = ONE flattened QA  (FANToM prompts each QA separately — a per-item protocol).
  process_results  scores the QA (vendored scorers, incl. the RoBERTa embedder for
                   BeliefQ[Dist.]) and emits a rich payload under EVERY report metric.
  aggregation      (one custom !function per metric) rebuilds a DataFrame from the
                   emitted payloads and runs vendored score_and_analyze for both the
                   `inaccessible` (FANToM) and `accessible` (control) scenarios.

v1 scope: short context, direct (non-CoT), aggregation_target = set.
"""

import functools
import hashlib
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

import datasets
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANDOM_SEED = 99  # ← eval_fantom.py:29 — drives the MC-belief option shuffle
_FANTOM_URL = "https://storage.googleapis.com/ai2-mosaic-public/projects/fantom/fantom.tar.gz"
_FANTOM_SHA256 = "1d08dfa0ea474c7f83b9bc7e3a7b466eab25194043489dd618b4c5223e1253a4"
_CACHE_DIR = Path.home() / ".cache" / "fantom"   # writable; NOT benchmarks/ (read-only)
_VERSION = "1.0"

CONVERSATION_INPUT_TYPE = "short"   # v1: short only
AGGREGATION_TARGET = "set"          # v1: set-level ALL/ALL* (README table setting)


# ---------------------------------------------------------------------------
# LOADER — download + unzip (vendored from task/dataset_loader.py, retargeted)
# ---------------------------------------------------------------------------

def _download_and_unzip():
    """Ensure fantom_v1.json exists under the cache dir; download+unpack once.
    Mirrors dataset_loader.build_data (hash-checked tar.gz → fantom/fantom_v1.json)."""
    target_dir = _CACHE_DIR / "fantom"
    json_path = target_dir / "fantom_v1.json"
    built_marker = target_dir / ".built"
    if json_path.exists() and built_marker.exists():
        return json_path

    import requests  # local import: only needed on first download

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(_FANTOM_URL, stream=True)
    data = b""
    for chunk in resp.iter_content(chunk_size=1024 * 1024 * 10):
        data += chunk
    if hashlib.sha256(data).hexdigest() != _FANTOM_SHA256:
        raise RuntimeError("FANToM download hash mismatch")
    tgz = _CACHE_DIR / "fantom.tar.gz"
    tgz.write_bytes(data)
    with tarfile.open(tgz) as tar:
        tar.extractall(target_dir)
    tgz.unlink()
    built_marker.write_text(_VERSION)
    return json_path


# ---------------------------------------------------------------------------
# RESHAPER + PROMPT — flatten sets → one doc per QA  (vendored setup_fantom)
# MC-belief options are shuffled ONCE here, sequentially, under seed 99 — exactly
# as eval_fantom seeds at module load and consumes the RNG in setup_fantom order.
# ---------------------------------------------------------------------------

def _set_beliefQA_multiple_choices(qa):
    """Vendored from eval_fantom.set_beliefQA_multiple_choices (uses module `random`)."""
    option_a = qa["wrong_answer"]
    option_b = qa["correct_answer"]
    answer_goes_last = random.choice([True, False])
    if answer_goes_last:
        choices = [option_a, option_b]
        answer = 1
    else:
        choices = [option_b, option_a]
        answer = 0
    option_letters = ["(" + chr(x) + ")" for x in range(ord("a"), len(choices) + ord("a"))]
    choices_text = ""
    for letter, option in zip(option_letters, choices):
        choices_text += "{} {}\n".format(letter, option)
    return choices_text, answer


def _flatten(df):
    """Vendored from eval_fantom.setup_fantom (conversation_input_type='short').
    Returns a list of QA dicts, each becoming one lm-eval doc."""
    random.seed(RANDOM_SEED)  # ← reproduce the shuffle deterministically, once
    qas = []
    for _, _set in df.iterrows():
        context = _set["short_context"].strip()
        set_id = _set["set_id"]
        fact_q = _set["factQA"]["question"]
        fact_a = _set["factQA"]["correct_answer"]

        # Fact Question
        fact = dict(_set["factQA"])
        fact["context"] = context
        fact["input_text"] = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
        fact["set_id"] = set_id
        qas.append(fact)

        for _belief_qa in _set["beliefQAs"]:
            belief = dict(_belief_qa)
            belief["context"] = context
            belief["input_text"] = "{}\n\nQuestion: {}\nAnswer:".format(context, belief["question"])
            belief["set_id"] = set_id
            qas.append(belief)

            # Multiple-choice belief (shuffle consumed here, in order)
            mc = dict(_belief_qa)
            choices_text, answer = _set_beliefQA_multiple_choices(mc)
            mc_question = "{}\n{}\n\nChoose an answer from above:".format(
                _belief_qa["question"], choices_text.strip()
            )
            mc["question"] = mc_question
            mc["question_type"] = mc["question_type"] + ":multiple-choice"
            mc["choices_text"] = choices_text
            mc["correct_answer"] = answer
            mc["context"] = context
            mc["input_text"] = "{}\n\nQuestion: {}".format(context, mc_question)
            mc["set_id"] = set_id
            qas.append(mc)

        # Answerability List
        alist = dict(_set["answerabilityQA_list"])
        alist["fact_question"] = fact_q
        alist["context"] = context
        alist["input_text"] = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(
            context, fact_q, alist["question"]
        )
        alist["set_id"] = set_id
        qas.append(alist)

        # Answerability Binary
        for _ab in _set["answerabilityQAs_binary"]:
            ab = dict(_ab)
            ab["fact_question"] = fact_q
            ab["context"] = context
            ab["input_text"] = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(
                context, fact_q, ab["question"]
            )
            ab["set_id"] = set_id
            qas.append(ab)

        # Info-Accessibility List
        ilist = dict(_set["infoAccessibilityQA_list"])
        ilist["fact_question"] = fact_q
        ilist["fact_answer"] = fact_a
        ilist["context"] = context
        ilist["input_text"] = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(
            context, fact_q, fact_a, ilist["question"]
        )
        ilist["set_id"] = set_id
        qas.append(ilist)

        # Info-Accessibility Binary
        for _ib in _set["infoAccessibilityQAs_binary"]:
            ib = dict(_ib)
            ib["fact_question"] = fact_q
            ib["fact_answer"] = fact_a
            ib["context"] = context
            ib["input_text"] = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(
                context, fact_q, fact_a, ib["question"]
            )
            ib["set_id"] = set_id
            qas.append(ib)

    return qas


@functools.lru_cache(maxsize=1)
def _build_docs():
    """Parse + flatten once. Heterogeneous QA dicts (correct_answer is str/int/list
    across types) are stored as a single JSON string `qa` to avoid Arrow schema
    clashes (the dyntom pattern); flat fields are exposed for doc_to_text / logging."""
    json_path = _download_and_unzip()
    df = pd.read_json(json_path)
    docs = []
    for qa in _flatten(df):
        docs.append({
            "input_text": qa["input_text"],
            "set_id": qa["set_id"],
            "question_type": qa["question_type"],
            "target": str(qa.get("correct_answer")),  # logging only
            "qa": json.dumps(qa, ensure_ascii=False),
        })
    return docs


def load(**kwargs):
    return {"train": datasets.Dataset.from_list(_build_docs())}


def doc_to_text(doc):
    return doc["input_text"]


# ---------------------------------------------------------------------------
# Per-QA scorers — vendored verbatim from eval_fantom.py (self → module funcs)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_embedder():
    """Lazy RoBERTa-large singleton (BeliefQ[Dist.] cosine). Heavy: loads only when
    a belief-distance QA is actually scored."""
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer("sentence-transformers/all-roberta-large-v1").to(device)


def compute_f1(ground_truth, model_response):
    ground_truth = ground_truth.split()
    model_response = model_response.split()
    common = Counter(ground_truth) & Counter(model_response)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(model_response)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def evaluate_belief_q(qa, model_response):
    from sklearn.metrics.pairwise import cosine_similarity
    embedder = _get_embedder()
    wrong_tom_view = qa["wrong_answer"]
    wrong_emb = embedder.encode(wrong_tom_view)
    personx_emb = embedder.encode(qa["correct_answer"])
    resp_emb = embedder.encode(model_response)
    sim_wrong = cosine_similarity(resp_emb.reshape(1, -1), wrong_emb.reshape(1, -1))[0][0]
    sim_personx = cosine_similarity(resp_emb.reshape(1, -1), personx_emb.reshape(1, -1))[0][0]
    if sim_wrong >= sim_personx:
        return False, compute_f1(wrong_tom_view, model_response)
    return True, compute_f1(qa["correct_answer"], model_response)


def evaluate_mc_belief_q(qa, model_response):
    int_to_alphabet = {0: "a", 1: "b", 2: "c", 3: "d"}
    answer = int_to_alphabet[qa["correct_answer"]]
    response = model_response.lower()
    return (
        response.startswith("(" + answer + ")")
        or response.startswith(answer + ")")
        or response.startswith(answer + ".")
        or response.startswith(answer + ":")
        or response.startswith(answer + ",")
        or "({})".format(answer) in response
        or answer == response
    )


def evaluate_list_q(qa, model_response):
    excluded_aware_character = False
    included_unaware_character = False
    for character in qa["correct_answer"]:
        if character.lower() not in model_response.lower():
            excluded_aware_character = True
            break
    for character in qa["wrong_answer"]:
        if character.lower() in model_response.lower():
            included_unaware_character = True
            break
    return (
        not (excluded_aware_character or included_unaware_character),
        excluded_aware_character,
        included_unaware_character,
    )


def map_binary_answer_to_int(model_response):
    a = model_response.lower().strip("'").strip('"')
    if (" yes," in a or " yes " in a or a.startswith("yes") or " yes." in a
            or " knows " in a or a.lower().startswith("true")):
        return 1
    elif (" no," in a or " no " in a or a.startswith("no") or " no." in a
          or " does not know " in a or " doesn't know " in a or a.lower().startswith("false")):
        return 0
    return -1


def evaluate_fact_q(qa, model_response):
    return compute_f1(qa["correct_answer"].lower(), model_response.lower())


def yesno_to_int(yesno_str):
    return {"yes": 1, "no": 0, "no:long": 0, "error": -1}[yesno_str]


def parse_response(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    elif "Choose an answer from above:" in response:
        response = response.split("Choose an answer from above:")[-1].strip()
    return response


# ---------------------------------------------------------------------------
# process_results — score one QA (vendored evaluate_response branching) and emit
# the SAME rich payload under every report metric (each agg consumes the full list).
# ---------------------------------------------------------------------------

def process_results(doc, results):
    qa = json.loads(doc["qa"])
    pred = parse_response(results[0])
    qt = qa["question_type"]

    payload = {
        "set_id": qa["set_id"],
        "question_type": qt,
        "correct_answer": qa.get("correct_answer"),
        "missed_info_accessibility": qa.get("missed_info_accessibility"),
        "tom_type": qa.get("tom_type"),
        "question": qa.get("question"),
        "prediction": pred,
        "word_overlap": None,
        "binarized_model_answer": None,
        "excluded_aware_character": None,
        "included_unaware_character": None,
    }

    if qt.startswith("tom:belief:"):
        if qt.endswith(":multiple-choice"):
            result = bool(evaluate_mc_belief_q(qa, pred))
        else:
            result, word_overlap = evaluate_belief_q(qa, pred)
            result = bool(result)
            payload["word_overlap"] = word_overlap
    elif qt.endswith(":list"):
        result, excl, incl = evaluate_list_q(qa, pred)
        result = bool(result)
        payload["excluded_aware_character"] = excl
        payload["included_unaware_character"] = incl
    elif qt.endswith(":binary"):
        _bin = map_binary_answer_to_int(pred)
        result = bool(yesno_to_int(qa["correct_answer"]) == _bin)
        payload["binarized_model_answer"] = _bin
    elif qt.startswith("fact"):
        result = float(evaluate_fact_q(qa, pred))
    else:
        raise NotImplementedError(qt)

    payload["result"] = result
    return {name: payload for name, _ in METRICS}


# ---------------------------------------------------------------------------
# Corpus report — vendored run_reports + score_and_analyze (headline scalars).
# Returns BOTH the `inaccessible` (FANToM) and `accessible` (control) numbers.
# ---------------------------------------------------------------------------

_F1 = None
_REPORT_COLS = [
    "set_id", "question_type", "correct_answer", "missed_info_accessibility",
    "result", "word_overlap", "binarized_model_answer", "prediction",
]


def _f1_metric():
    global _F1
    if _F1 is None:
        import evaluate
        _F1 = evaluate.load("f1")
    return _F1


def _score_scenario(df, target_scenario, agg_col):
    """Vendored headline portion of eval_fantom.score_and_analyze (one scenario)."""
    report = {}
    tom_df = df[df["question_type"].str.startswith("tom")].copy()
    target_df = tom_df[tom_df["missed_info_accessibility"] == target_scenario].copy()

    if target_scenario == "accessible":
        # Keep only sets whose every tom question is `accessible` (else ALL/ALL* inflate)
        set_ids = target_df["set_id"].unique()
        target_sets = [
            sid for sid in set_ids
            if tom_df[tom_df["set_id"] == sid]["missed_info_accessibility"].eq(target_scenario).all()
        ]
    else:
        target_sets = target_df["set_id"].unique()

    s = target_scenario
    report[s + ":set:ALL*"] = (
        target_df[target_df["set_id"].isin(target_sets)].groupby(agg_col)["result"].all().mean()
    )
    qfa = ["tom:belief:" + s + ":multiple-choice", "tom:answerability:list",
           "tom:answerability:binary", "tom:info_accessibility:list", "tom:info_accessibility:binary"]
    report[s + ":set:ALL"] = (
        target_df[target_df["question_type"].isin(qfa) & target_df["set_id"].isin(target_sets)]
        .groupby(agg_col)["result"].all().mean()
    )
    report[s + ":belief:multiple-choice"] = (
        target_df[target_df["question_type"].str.endswith(":multiple-choice")]["result"].mean()
    )
    report[s + ":belief:distance"] = (
        target_df[target_df["question_type"] == "tom:belief:" + s]["result"].mean()
    )
    report[s + ":belief_true_word-f1"] = (
        target_df[(target_df["question_type"] == "tom:belief:" + s) & (target_df["result"] == True)]
        ["word_overlap"].mean()
    )
    report[s + ":answerability:set:ALL"] = (
        target_df[target_df["question_type"].str.startswith("tom:answerability")]
        .groupby(agg_col)["result"].all().mean()
    )
    report[s + ":answerability:list"] = (
        target_df[target_df["question_type"] == "tom:answerability:list"]["result"].mean()
    )
    ans_pred = target_df[target_df["question_type"] == "tom:answerability:binary"]["binarized_model_answer"].to_list()
    ans_ref = target_df[target_df["question_type"] == "tom:answerability:binary"]["correct_answer"].map(yesno_to_int).to_list()
    report[s + ":answerability:binary-f1"] = (
        _f1_metric().compute(predictions=ans_pred, references=ans_ref, pos_label=0, average="weighted")["f1"]
        if ans_pred else float("nan")
    )
    report[s + ":info_accessibility:set:ALL"] = (
        target_df[target_df["question_type"].str.startswith("tom:info_accessibility")]
        .groupby(agg_col)["result"].all().mean()
    )
    report[s + ":info_accessibility:list"] = (
        target_df[target_df["question_type"] == "tom:info_accessibility:list"]["result"].mean()
    )
    acc_pred = target_df[target_df["question_type"] == "tom:info_accessibility:binary"]["binarized_model_answer"].to_list()
    acc_ref = target_df[target_df["question_type"] == "tom:info_accessibility:binary"]["correct_answer"].map(yesno_to_int).to_list()
    report[s + ":info_accessibility:binary-f1"] = (
        _f1_metric().compute(predictions=acc_pred, references=acc_ref, pos_label=0, average="weighted")["f1"]
        if acc_pred else float("nan")
    )
    report["fact_word-f1"] = df[df["question_type"].str.startswith("fact")]["result"].mean()

    for k, v in report.items():
        if isinstance(v, float):
            report[k] = round(v, 3) * 100
    return report


def _score_report(payloads):
    """Vendored run_reports: build df, drop short-input `no:long` binaries, then
    score both scenarios. Recomputed per metric (cheap; headline-only scoring)."""
    df = pd.DataFrame(list(payloads))
    for col in _REPORT_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    # short input: drop binary questions with no:long answer (run_reports)
    df = df.drop(
        df[(df["question_type"].str.endswith(":binary")) & (df["correct_answer"] == "no:long")].index
    )
    agg_col = AGGREGATION_TARGET + "_id"  # v1: "set_id"
    if agg_col not in df.columns:
        # run_reports derives these from set_id; we only need set_id for v1
        df["conversation_id"] = df["set_id"].map(lambda x: x.split("-")[0])
        df["part_id"] = df["set_id"].map(lambda x: "-".join(x.split("-")[:2]))
    report = _score_scenario(df, "inaccessible", agg_col)
    control = _score_scenario(df, "accessible", agg_col)
    return {**report, **control}


# ---------------------------------------------------------------------------
# Metric registry — (lm-eval metric name, score_and_analyze report key).
# process_results emits the payload under every name; each agg returns its key.
# ---------------------------------------------------------------------------

METRICS = [
    # inaccessible (the FANToM task)
    ("all_star",                 "inaccessible:set:ALL*"),
    ("all",                      "inaccessible:set:ALL"),
    ("belief_choice",            "inaccessible:belief:multiple-choice"),
    ("belief_dist",              "inaccessible:belief:distance"),
    ("belief_tokenf1",           "inaccessible:belief_true_word-f1"),
    ("answerability_all",        "inaccessible:answerability:set:ALL"),
    ("answerability_list",       "inaccessible:answerability:list"),
    ("answerability_binary_f1",  "inaccessible:answerability:binary-f1"),
    ("infoaccess_all",           "inaccessible:info_accessibility:set:ALL"),
    ("infoaccess_list",          "inaccessible:info_accessibility:list"),
    ("infoaccess_binary_f1",     "inaccessible:info_accessibility:binary-f1"),
    ("fact_tokenf1",             "fact_word-f1"),
    # accessible (the mandated control task)
    ("control_all_star",                "accessible:set:ALL*"),
    ("control_all",                     "accessible:set:ALL"),
    ("control_belief_choice",           "accessible:belief:multiple-choice"),
    ("control_belief_dist",             "accessible:belief:distance"),
    ("control_belief_tokenf1",          "accessible:belief_true_word-f1"),
    ("control_answerability_all",       "accessible:answerability:set:ALL"),
    ("control_answerability_list",      "accessible:answerability:list"),
    ("control_answerability_binary_f1", "accessible:answerability:binary-f1"),
    ("control_infoaccess_all",          "accessible:info_accessibility:set:ALL"),
    ("control_infoaccess_list",         "accessible:info_accessibility:list"),
    ("control_infoaccess_binary_f1",    "accessible:info_accessibility:binary-f1"),
]


def _make_agg(report_key):
    def _agg(items):
        return _score_report(items).get(report_key, float("nan"))
    return _agg


for _name, _key in METRICS:
    globals()["agg_" + _name] = _make_agg(_key)
