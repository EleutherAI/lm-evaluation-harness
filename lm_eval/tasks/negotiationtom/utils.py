from __future__ import annotations

import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import datasets


AGENTS = ("agent_1", "agent_2")
SLOT_ORDER = ("high", "medium", "low")
ITEM_PROMPT_CHOICES = ["Not given", "Water", "Food", "Firewood"]
ITEM_VALUE_TO_INDEX = {
    "Not Given": 0,
    "Not given": 0,
    "Water": 1,
    "Food": 2,
    "Firewood": 3,
}
INTENT_PROMPT_CHOICES = [
    ("A", "Intents to build a rapport with the opponent"),
    ("B", "Intents to show empathy with the opponent"),
    ("C", "Intents to promote coordination with the opponent"),
    ("D", "Intents to callout to fairness"),
    ("E", "Intents to undermine the requirements of the opponent"),
    ("F", "Intents to discover the preference order of the opponent"),
    ("G", "Intents to describe a need for an item"),
    ("H", "Intents to point out they do not need an item"),
    ("I", "No clear intention in the utterance"),
]
INTENT_LETTER_TO_LABEL = {
    "A": "Build-Rapport",
    "B": "Show-Empathy",
    "C": "Promote-Coordination",
    "D": "Callout-Fairness",
    "E": "Undermine-Requirements",
    "F": "Discover-Preference",
    "G": "Describe-Need",
    "H": "No-Need",
    "I": "No-Intention",
}
INTENT_ALIAS_TO_LABEL = {
    "build-rapport": "Build-Rapport",
    "show-empathy": "Show-Empathy",
    "promote-coordination": "Promote-Coordination",
    "callout-fairness": "Callout-Fairness",
    "undermine-requirements": "Undermine-Requirements",
    "discover-preference": "Discover-Preference",
    "describe-need": "Describe-Need",
    "no-need": "No-Need",
    "no-intention": "No-Intention",
    "build rapport": "Build-Rapport",
    "small talk": "Build-Rapport",
    "show empathy": "Show-Empathy",
    "empathy": "Show-Empathy",
    "promote coordination": "Promote-Coordination",
    "coordination": "Promote-Coordination",
    "callout to fairness": "Callout-Fairness",
    "vouch fairness": "Callout-Fairness",
    "undermine the requirements of the opponent": "Undermine-Requirements",
    "undervalue partner": "Undermine-Requirements",
    "discover the preference order of the opponent": "Discover-Preference",
    "elicit pref": "Discover-Preference",
    "describe a need for an item": "Describe-Need",
    "self need": "Describe-Need",
    "point out they do not need an item": "No-Need",
    "no need": "No-Need",
    "no clear intention in the utterance": "No-Intention",
    "non strategic": "No-Intention",
}
LETTER_RE = re.compile(r"\b([A-I])\b")
NORMALIZE_RE = re.compile(r"\s+")


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "benchmarks" / "NegotiationToM"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find benchmarks/NegotiationToM from the task directory."
    )


@lru_cache(maxsize=1)
def _load_data() -> list[dict[str, Any]]:
    root = _repo_root()
    zip_path = root / "NegotiationToM.zip"
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("NegotiationToM.json", pwd=b"NegotiationToM") as handle:
            return json.load(io.TextIOWrapper(handle, encoding="utf-8"))


def _parse_labels(raw: str) -> set[str]:
    return {part.strip() for part in str(raw).split(",") if part.strip()}


def _render_dialogue(dialogue: list[str]) -> str:
    return "\n".join(dialogue)


def _normalize(text: str) -> str:
    return NORMALIZE_RE.sub(" ", (text or "").strip().lower())


def _item_choice_block() -> str:
    return "\n".join(
        f"{letter}.{choice}" for letter, choice in zip(("A", "B", "C", "D"), ITEM_PROMPT_CHOICES)
    )


def _intent_choice_block() -> str:
    return "\n".join(f"{letter}.{label}" for letter, label in INTENT_PROMPT_CHOICES)


def _agent_label(agent: str) -> str:
    return agent.replace("_", " ").title()


def _other_agent(agent: str) -> str:
    return "Agent 2" if agent == "agent_1" else "Agent 1"


def _row_agent(agent: str) -> str:
    return agent.replace("_", "")


def _build_base_context(dialogue: str) -> str:
    return (
        "Background: Here is a negotiation conversation for a camping trip. "
        "There are two agents who own some basic supplies and negotiate with each other "
        "to split the additional food packages, water bottles, and firewood to make "
        "their camping trip even better. Each of these items will be of either High, "
        "Medium or Low priority for these two agents. Each of the additional items only "
        "has an available quantity of 3.\nDialogue History:\n"
        f"{dialogue}\n"
    )


def _build_desire_or_belief_prompt(row: dict[str, Any], agent: str, mode: str) -> str:
    dialogue = _render_dialogue(row["dialogue"])
    agent_text = _agent_label(agent)
    other_text = _other_agent(agent)
    intro = _build_base_context(dialogue)
    if mode == "belief":
        question_intro = (
            'Please answer the following three questions using "A", "B", "C", "D".\n'
            f"Question1: Based on the dialogue, what is the high preference for items {agent_text} thinks {other_text} is?\n"
            f"Question2: Based on the dialogue, what is the medium preference for items {agent_text} thinks {other_text} is?\n"
            f"Question3: Based on the dialogue, what is the low preference for items {agent_text} thinks {other_text} is?\n"
        )
    elif mode == "desire":
        question_intro = (
            'Please answer the following three questions using "A", "B", "C", "D" without any explanation.\n'
            f"Question1: What is {agent_text}'s high preference for items based on the dialogue history?\n"
            f"Question2: What is {agent_text}'s medium preference for items based on the dialogue history?\n"
            f"Question3: What is {agent_text}'s low preference for items based on the dialogue history?\n"
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return f"{intro}{question_intro}{_item_choice_block()}\nAnswer:"


def _build_intent_prompt(row: dict[str, Any], utterance: str, speaker: str) -> str:
    dialogue = _render_dialogue(row["dialogue"])
    agent_text = _agent_label(speaker)
    intro = _build_base_context(dialogue)
    return (
        f"{intro}"
        f"Please imagine that you are {agent_text} and infer your strategies expressed in '{utterance}' "
        f"from the dialogue history. "
        f"Select one or more strategies (i.e., \"A\", \"B\", \"C\",...,\"I\") from the following choices without any explanation.\n"
        f"{_intent_choice_block()}\n"
        "Answer:"
    )


def _build_desire_or_belief_docs(mode: str) -> list[dict[str, Any]]:
    data = _load_data()
    docs: list[dict[str, Any]] = []
    for row in data:
        for agent in AGENTS:
            fields = [f"{_row_agent(agent)}_{mode}_{slot}" for slot in SLOT_ORDER]
            values = [row[field] for field in fields]
            if any(value == "None" for value in values):
                continue
            for slot, field in zip(SLOT_ORDER, fields):
                gold = ITEM_VALUE_TO_INDEX[row[field]]
                docs.append(
                    {
                        "dialogue_id": row["dialogue_id"],
                        "agent": agent,
                        "slot": slot,
                        "mode": mode,
                        "dialogue": row["dialogue"],
                        "prompt": _build_desire_or_belief_prompt(row, agent, mode),
                        "choices": ["A", "B", "C", "D"],
                        "gold": gold,
                    }
                )
    return docs


def _build_intent_docs() -> list[dict[str, Any]]:
    data = _load_data()
    docs: list[dict[str, Any]] = []
    for row in data:
        for utterance_idx, (utterance_key, speaker_key) in enumerate(
            (("utterance1_intent", "utterance1_agent"), ("utterance2_intent", "utterance2_agent"))
        ):
            speaker = row[speaker_key]
            if speaker == "None":
                continue
            if utterance_idx >= len(row["dialogue"]):
                continue
            utterance = row["dialogue"][utterance_idx]
            docs.append(
                {
                    "dialogue_id": row["dialogue_id"],
                    "utterance_idx": utterance_idx,
                    "dialogue": row["dialogue"],
                    "utterance": utterance,
                    "gold_labels": row[utterance_key],
                    "prompt": _build_intent_prompt(row, utterance, speaker),
                }
            )
    return docs


def load_dataset(mode: str = "desire", **_: Any) -> dict[str, datasets.Dataset]:
    if mode == "intention":
        return {"test": datasets.Dataset.from_list(_build_intent_docs())}
    if mode not in {"desire", "belief"}:
        raise ValueError(f"Unsupported mode: {mode}")
    return {"test": datasets.Dataset.from_list(_build_desire_or_belief_docs(mode))}


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["prompt"]


def _normalize_text(text: str) -> str:
    return _normalize(text)


def _tokenize_labels(text: str) -> list[str]:
    normalized = _normalize_text(text)
    labels: list[str] = []
    seen: set[str] = set()

    for letter in LETTER_RE.findall((text or "").upper()):
        label = INTENT_LETTER_TO_LABEL.get(letter)
        if label and label not in seen:
            labels.append(label)
            seen.add(label)

    for alias, label in INTENT_ALIAS_TO_LABEL.items():
        if alias in normalized and label not in seen:
            labels.append(label)
            seen.add(label)

    return labels


def process_intent_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    pred = set(_tokenize_labels(results[0] if results else ""))
    gold = _parse_labels(doc["gold_labels"])
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
