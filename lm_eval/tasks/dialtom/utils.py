from __future__ import annotations

import csv
import io
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any

import datasets


DATASET_NAMES = ("MI", "ESC", "PFG")
TASK_DESCRIPTIONS = {
    "MI": "Counseling Session",
    "ESC": "Emotional Support Conversation",
    "PFG": "Persuasion Conversation",
}
AGENTS = {
    "MI": ("Counselor", "Client"),
    "ESC": ("Supporter", "Seeker"),
    "PFG": ("Persuader", "Persuadee"),
}
INTENT_LABELS = [
    "Build-Rapport",
    "Callout-Fairness",
    "Describe-Need",
    "Discover-Preference",
    "No-Need",
    "No-Intention",
    "Promote-Coordination",
    "Show-Empathy",
    "Undermine-Requirements",
]
CHOICE_RE = re.compile(r"\b([A-E])\b")
TOKEN_SPLIT = re.compile(r"[,\n;/|]+")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "benchmarks" / "DialToM"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find benchmarks/DialToM from the task directory.")


def _load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_domain_rows(domain: str, mode: str) -> list[dict[str, Any]]:
    path = _repo_root() / "data" / f"{domain}_{mode}_verified.json"
    rows = _load_json(path)
    for row in rows:
        row["dataset"] = domain
    return rows


def _build_retrospective_doc(row: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    domain = row["dataset"]
    task_desc = TASK_DESCRIPTIONS[domain]
    agent1, agent2 = AGENTS[domain]
    state = row["state"]
    correct_letter = row["correct_option"][state]
    options = list(row["options"][state].items())
    rng.shuffle(options)
    choices = [text for _, text in options]
    gold = next(idx for idx, (letter, _) in enumerate(options) if letter == correct_letter)
    options_text = "\n".join(f"{chr(65 + idx)}: {choice}" for idx, choice in enumerate(choices))
    ctx = "\n".join(row["ctx"])
    prompt = (
        "You are an expert in Theory of Mind reasoning.\n\n"
        "Task:\n"
        f"You will be provided with conversation between two agents {agent1} and {agent2} "
        f"engaging in a {task_desc} session on the topic of {row['topic']}.\n\n"
        f"Your goal is to correctly infer {agent2}'s {state} state, based on the above conversation. "
        "You will be provided with a set of options, and you need to choose the most appropriate one "
        f"that reflects the {state} state.\n\n"
        "The correct option must be consistent with the provided conversation context.\n\n"
        "Conversation Context:\n"
        f"{ctx}\n\n"
        "Mental State Options:\n"
        f"{options_text}\n\n"
        "Instruction\n"
        "Output only the letter of the correct option (e.g., \"A\", \"B\", \"C\", or \"D\"). Do not add explanations or other verbosity.\n"
        "Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE. ONLY OUTPUT THE OPTION AND NOTHING ELSE. YOUR OUTPUT SHOULD STRICTLY BE ONE OF A, B, C, or D.\n\n"
        "Answer:\n"
    )
    return {
        "prompt": prompt,
        "choices": list("ABCD"),
        "gold": gold,
        "dataset": domain,
        "state": state,
        "topic": row["topic"],
    }


def _build_prospective_doc(
    row: dict[str, Any],
    rng: random.Random,
    exp: str,
    domain_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    domain = row["dataset"]
    agent1, agent2 = AGENTS[domain]
    all_mental_states = []
    for state in ("Belief", "Desires", "Intentions", "Emotions", "Knowledge", "Trust"):
        state_text = row["options"][state][row["correct_option"][state]]
        all_mental_states.append(f"{state}: {state_text}")

    if exp == "easy":
        other_rows = [item for item in domain_rows if item["id"] != row["id"]]
        sampled_ids = rng.sample([item["id"] for item in other_rows], 3)
        lookup = {item["id"]: item for item in domain_rows}
        distractors = ["\n".join(lookup[item_id]["ctx"][:4]) for item_id in sampled_ids]
    else:
        distractors = list(row["distractors"])

    choices = [row["correct_action"], *distractors]
    rng.shuffle(choices)
    gold = choices.index(row["correct_action"])
    letters = list("ABCD")
    if exp == "NOTA":
        choices.append("NOTA.")
        letters.append("E")
        instructions = (
            "Output only the letter of the correct option (e.g., \"A\", \"B\", \"C\", \"D\", or \"E\"). Do not add explanations or other verbosity.\n"
            "Your output should be strictly one of: A, B, C, D, E. NO FORMATTING NEEDS TO BE DONE. Choose Option E if you think none of the options are correct continuations."
        )
    elif exp == "CoT":
        instructions = (
            "Output only the letter of the correct option (e.g., \"A\", \"B\", \"C\", or \"D\"). Do not add explanations or other verbosity.\n"
            "Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE.\n\n"
            "Let's think step-by-step."
        )
    else:
        instructions = (
            "Output only the letter of the correct option (e.g., \"A\", \"B\", \"C\", or \"D\"). Do not add explanations or other verbosity.\n"
            "Your output should be strictly one of: A, B, C, D. NO FORMATTING NEEDS TO BE DONE."
        )

    options_text = "\n".join(
        f"{chr(65 + idx)}: {choice.replace('agent 1', agent1).replace('agent 2', agent2)}"
        for idx, choice in enumerate(choices)
    )
    prompt = (
        "You are an expert in Theory of Mind reasoning.\n\n"
        "Task:\n"
        "You will be provided with internal Mental State profile (Belief, Desire, Intention, Emotion, Knowledge, Trust) of client during the conversation.\n\n"
        "Your goal is to identify which of the candidate conversation segments is the most plausible continuation of this conversation.\n"
        "The correct option must be consistent with the provided Mental States of the client.\n\n"
        f"Mental state of {agent2}\n"
        + "\n".join(all_mental_states)
        + "\n\nCandidate Conversation Segments\n"
        + f"{options_text}\n\nInstruction\n{instructions}\n\nAnswer:\n"
    )
    return {
        "prompt": prompt,
        "choices": letters,
        "gold": gold,
        "dataset": domain,
        "exp": exp,
    }


def _build_written_docs() -> list[dict[str, Any]]:
    root = _repo_root()
    refs_path = root / "data" / "written_inference.csv"
    combined = json.loads((root / "data" / "combined_written_data.json").read_text(encoding="utf-8"))
    refs: dict[tuple[str, str, str], list[str]] = {}
    with refs_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["mental_state"], row["sub_id"])
            refs.setdefault(key, []).append(row["inferences"])

    docs: list[dict[str, Any]] = []
    for (dataset, mental_state, sub_id), ref_list in sorted(refs.items()):
        curr_data = combined[dataset][sub_id]
        ctx = "\n".join(
            line.replace("agent 1", AGENTS[dataset][0])
            .replace("agent 2", AGENTS[dataset][1])
            .replace("agent1", AGENTS[dataset][0])
            .replace("agent2", AGENTS[dataset][1])
            for line in curr_data["ctx"]
        )
        docs.append(
            {
                "dataset": dataset,
                "mental_state": mental_state,
                "sub_id": sub_id,
                "context": ctx,
                "refs": ref_list,
            }
        )
    return docs


def load_dataset(mode: str = "retrospective", exp: str = "normal") -> dict[str, datasets.Dataset]:
    rng = random.Random(42)
    if mode == "written":
        return {"test": datasets.Dataset.from_list(_build_written_docs())}

    records: list[dict[str, Any]] = []
    for domain in DATASET_NAMES:
        domain_rows = _load_domain_rows(domain, mode)
        if mode == "retrospective":
            for row in domain_rows:
                records.append(_build_retrospective_doc(row, rng))
        elif mode == "prospective":
            for row in domain_rows:
                records.append(_build_prospective_doc(row, rng, exp, domain_rows))
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    return {"test": datasets.Dataset.from_list(records)}


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["prompt"]


def process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    pred = _extract_choice(results[0] if results else "", doc["choices"])
    gold = doc["choices"][doc["gold"]]
    return {"acc": 1.0 if pred == gold else 0.0}


def _extract_choice(text: str, choices: list[str]) -> str:
    stripped = (text or "").strip().upper()
    if stripped in choices:
        return stripped
    match = CHOICE_RE.search(stripped)
    if match and match.group(1) in choices:
        return match.group(1)
    return ""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize_labels(text: str) -> list[str]:
    normalized = _normalize_text(text)
    labels = []
    for label in INTENT_LABELS:
        if label.lower() in normalized:
            labels.append(label)
    return labels


def _max_over_refs(scores: list[float]) -> float:
    return max(scores) if scores else 0.0


def _score_bleu(pred: str, refs: list[str]) -> float:
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    return _max_over_refs([bleu.sentence_score(pred, [ref]).score for ref in refs])


def _score_rouge_l(pred: str, refs: list[str]) -> float:
    from rouge import Rouge

    rouge = Rouge()
    return _max_over_refs([rouge.get_scores(pred, [ref])[0]["rouge-l"]["f"] for ref in refs])


_BERT_SCORER = None


def _get_bert_scorer():
    global _BERT_SCORER
    if _BERT_SCORER is None:
        import bert_score

        _BERT_SCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    return _BERT_SCORER


def _score_bertscore(pred: str, refs: list[str]) -> float:
    scorer = _get_bert_scorer()
    preds = [pred] * len(refs)
    _, _, f1 = scorer.score(preds, refs, verbose=False)
    return float(f1.max().item()) if len(f1) else 0.0


def process_written_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    pred = (results[0] if results else "").strip()
    refs = [ref.strip() for ref in doc["refs"]]
    return {
        "bleu": _score_bleu(pred, refs),
        "rouge_l": _score_rouge_l(pred, refs),
        "bertscore": _score_bertscore(pred, refs),
    }


def process_intent_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    pred = set(_tokenize_labels(results[0] if results else ""))
    gold = set(label.strip() for label in str(doc["gold_labels"]).split(",") if label.strip())
    return {
        "micro_f1": (gold, pred),
        "macro_f1": (gold, pred),
        "exact_match": 1.0 if pred == gold else 0.0,
    }


def micro_f1_agg(items: list[tuple[set[str], set[str]]]) -> float:
    tp = fp = fn = 0
    for gold, pred in items:
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    if tp == fp == fn == 0:
        return 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def macro_f1_agg(items: list[tuple[set[str], set[str]]]) -> float:
    if not items:
        return 0.0
    labels = sorted({label for gold, pred in items for label in gold | pred})
    if not labels:
        return 0.0

    def _label_f1(label: str) -> float:
        tp = fp = fn = 0
        for gold, pred in items:
            if label in gold and label in pred:
                tp += 1
            elif label not in gold and label in pred:
                fp += 1
            elif label in gold and label not in pred:
                fn += 1
        if tp == fp == fn == 0:
            return 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    return sum(_label_f1(label) for label in labels) / len(labels)


def intent_doc_to_text(doc: dict[str, Any]) -> str:
    return (
        "You are an expert in Theory of Mind reasoning.\n\n"
        "Task:\n"
        "You will be provided with a negotiation dialogue.\n\n"
        "Your goal is to identify the intentions expressed in this utterance from the dialogue.\n\n"
        f"Dialogue:\n{doc['dialogue']}\n\n"
        f"Utterance:\n{doc['utterance']}\n\n"
        "Available intention labels:\n"
        f"{', '.join(INTENT_LABELS)}\n\n"
        "Instruction\n"
        "Output all applicable labels separated by commas, and nothing else.\n\n"
        "Answer:\n"
    )
