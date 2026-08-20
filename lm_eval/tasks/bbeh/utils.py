# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# This evaluator is adapted from google-deepmind/bbeh at revision
# 80d12ca916b7158f22293fcf3144f4d3d854d4be.

"""Official answer normalization and scoring for BIG-Bench Extra Hard."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Sequence


FINAL_ANSWER_SUFFIX = (
    "Think step by step, and when you provide the final answer, please use the "
    'prefix "The answer is:" without any modification, and provide the answer '
    "directly, with no formatting, no bolding, and no markup. For instance: "
    '"The answer is: 42" or  "The answer is: yes". If the question is multiple '
    "choice with a single correct answer, the final answer must only be the "
    'letter corresponding to the correct answer. For example, "The answer is: '
    '(a)".'
)


def doc_to_text(doc: dict) -> str:
    """Append the fixed final-answer suffix used in the BBEH paper."""
    return f"{doc['input'].rstrip()}\n\n{FINAL_ANSWER_SUFFIX}"


def strip_latex(response: str) -> str:
    """Strip the wrappers handled by the official BBEH evaluator."""
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extract the final answer using the ordered official prefixes."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    answer = answer.removesuffix(".")
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Apply the official BBEH fuzzy equality rules."""
    if prediction == reference:
        return True

    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


def preprocess_sample(sample: str) -> str:
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    return prediction[0:-1] if prediction.endswith(".") else prediction


def preprocess_reference(reference: str) -> str:
    return reference.strip().lower().replace(", ", ",")


def evaluate_correctness(sample: str, reference: str) -> bool:
    """Return whether a generation matches a BBEH reference."""
    return fuzzy_match(preprocess_sample(sample), preprocess_reference(reference))


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Convert a model generation into the official per-example score."""
    return {"bbeh_acc": float(evaluate_correctness(results[0], doc["target"]))}


def harmonic_mean(task_accuracies: Sequence[float]) -> float:
    """Compute BBEH's adjusted headline harmonic mean across task accuracies.

    lm-eval's group layer currently supports only arithmetic mean aggregation.
    This helper is kept next to the evaluator so result exporters can compute
    the official headline value from the 23 per-task rows. The paper adds one
    percentage point to every task accuracy before taking the harmonic mean.
    """
    if not task_accuracies:
        raise ValueError("BBEH harmonic mean requires at least one task accuracy")
    if any(score < 0 for score in task_accuracies):
        raise ValueError("BBEH task accuracies must be non-negative")
    offset = 0.01
    return len(task_accuracies) / math.fsum(
        1.0 / (score + offset) for score in task_accuracies
    )
