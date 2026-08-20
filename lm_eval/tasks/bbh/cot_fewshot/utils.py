"""OLMES-compatible likelihood prompt and target for CoT BBH."""

import re


def bpb_text(doc: dict) -> str:
    # OLMES supplies ``perplexity_query`` for this path, which intentionally
    # removes the CoT cue and the generation task's few-shot context.
    return f"Q: {doc['input']}\nA:"


def bpb_target(doc: dict) -> str:
    match = re.search(r"(?<=answer is )(.*)(?=.)", doc["target"])
    return match[0] if match else doc["target"]
