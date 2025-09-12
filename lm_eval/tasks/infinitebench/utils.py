"""Utility functions for InfiniteBench evaluation."""

import json
import re
import string
from collections import Counter
from typing import Any, Dict, List, Union


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_passkey(text: str) -> str:
    """Extract passkey from model output."""
    # Look for patterns like "passkey is X" or "key: X" or just the key itself
    patterns = [
        r"passkey\s*(?:is|:)?\s*([A-Za-z0-9]+)",
        r"key\s*(?:is|:)?\s*([A-Za-z0-9]+)",
        r"answer\s*(?:is|:)?\s*([A-Za-z0-9]+)",
        r"^([A-Za-z0-9]+)$",
    ]

    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return text


def extract_number(text: str) -> str:
    """Extract number from model output."""
    # Look for numeric patterns
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[0]
    return text


def extract_code_output(text: str) -> str:
    """Extract code execution output from model response."""
    # Look for output patterns
    patterns = [
        r"output\s*(?:is|:)?\s*(.+)",
        r"result\s*(?:is|:)?\s*(.+)",
        r"returns?\s*(?:is|:)?\s*(.+)",
        r"^(.+)$",
    ]

    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return text


def passkey_accuracy(prediction: str, ground_truth: str) -> float:
    """Evaluate passkey retrieval accuracy."""
    pred = extract_passkey(prediction).strip()
    truth = ground_truth.strip()
    return 1.0 if pred == truth else 0.0


def number_string_accuracy(prediction: str, ground_truth: str) -> float:
    """Evaluate number string retrieval accuracy."""
    pred = extract_number(prediction)
    truth = extract_number(ground_truth)

    try:
        # Try numeric comparison
        pred_num = float(pred)
        truth_num = float(truth)
        return 1.0 if abs(pred_num - truth_num) < 0.001 else 0.0
    except (ValueError, TypeError):
        # Fall back to string comparison
        return 1.0 if pred == truth else 0.0


def kv_retrieval_accuracy(prediction: str, ground_truth: str) -> float:
    """Evaluate key-value retrieval accuracy."""
    pred = normalize_text(prediction)
    truth = normalize_text(ground_truth)
    return 1.0 if pred == truth else 0.0


def math_accuracy(
    prediction: str, ground_truth: str, tolerance: float = 0.001
) -> float:
    """Evaluate mathematical answer accuracy."""
    try:
        pred_num = float(extract_number(prediction))
        truth_num = float(extract_number(ground_truth))
        return 1.0 if abs(pred_num - truth_num) < tolerance else 0.0
    except (ValueError, TypeError):
        # If not numeric, do exact match
        return (
            1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0
        )


def code_execution_accuracy(prediction: str, ground_truth: str) -> float:
    """Evaluate code execution output accuracy."""
    pred = extract_code_output(prediction)
    truth = ground_truth.strip()

    # Try to parse as JSON for structured outputs
    try:
        pred_json = json.loads(pred)
        truth_json = json.loads(truth)
        return 1.0 if pred_json == truth_json else 0.0
    except (json.JSONDecodeError, ValueError):
        pass

    # Otherwise do normalized comparison
    return 1.0 if normalize_text(pred) == normalize_text(truth) else 0.0


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score for QA tasks."""

    def get_tokens(text):
        return normalize_text(text).split()

    pred_tokens = get_tokens(prediction)
    truth_tokens = get_tokens(ground_truth)

    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L score for summarization tasks."""

    def lcs_length(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[m][n]

    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(ground_truth).split()

    if not pred_tokens or not truth_tokens:
        return 0.0

    lcs_len = lcs_length(pred_tokens, truth_tokens)

    precision = lcs_len / len(pred_tokens) if pred_tokens else 0
    recall = lcs_len / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def multiple_choice_accuracy(prediction: str, ground_truth: str) -> float:
    """Evaluate multiple choice answer accuracy."""
    # Extract choice letter (A, B, C, D, etc.)
    pred_match = re.search(r"\b([A-Z])\b", prediction.upper())
    truth_match = re.search(r"\b([A-Z])\b", ground_truth.upper())

    if pred_match and truth_match:
        return 1.0 if pred_match.group(1) == truth_match.group(1) else 0.0

    # Fall back to normalized comparison
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0


# Task-specific metric mappings
TASK_METRICS = {
    "passkey": passkey_accuracy,
    "number_string": number_string_accuracy,
    "kv_retrieval": kv_retrieval_accuracy,
    "math_find": math_accuracy,
    "math_calc": math_accuracy,
    "code_run": code_execution_accuracy,
    "code_debug": qa_f1_score,
    "longbook_qa_eng": qa_f1_score,
    "longbook_qa_chn": qa_f1_score,
    "longbook_sum_eng": rouge_l_score,
    "longbook_choice_eng": multiple_choice_accuracy,
    "longdialogue_qa_eng": qa_f1_score,
}


def get_metric_for_task(task_name: str):
    """Get appropriate metric function for an InfiniteBench task."""
    task_name_lower = task_name.lower()
    for key, metric in TASK_METRICS.items():
        if key in task_name_lower:
            return metric
    # Default to F1 score
    return qa_f1_score


def process_results_gen(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """Process generation results for InfiniteBench tasks."""
    prediction = results[0] if results else ""

    # Get ground truth
    answer = doc.get("answer", doc.get("target", ""))
    if isinstance(answer, list):
        answer = answer[0] if answer else ""

    # Get task name to determine metric
    task_name = doc.get("task_name", doc.get("task", ""))
    metric_fn = get_metric_for_task(task_name)

    # Compute score
    score = metric_fn(prediction, str(answer))

    return {"score": score}


def create_prompt(doc: Dict[str, Any]) -> str:
    """Create prompt for InfiniteBench tasks."""
    context = doc.get("context", "")
    question = doc.get("question", doc.get("prompt", ""))
    task_type = doc.get("task_type", "")

    # Task-specific prompts
    if "passkey" in task_type.lower():
        prompt = f"""There is a hidden passkey in the following long text. Find and return only the passkey.

Text:
{context}

What is the passkey?
Answer:"""

    elif "code" in task_type.lower():
        prompt = f"""Read the following code and answer the question.

Code:
{context}

Question: {question}
Answer:"""

    elif "book" in task_type.lower() and "sum" in task_type.lower():
        prompt = f"""Summarize the following book excerpt.

Text:
{context}

Summary:"""

    elif "choice" in task_type.lower():
        prompt = f"""Read the following text and answer the multiple choice question.

Text:
{context}

{question}

Answer with only the letter of the correct choice:"""

    else:
        # Default QA format
        prompt = f"""Read the following text and answer the question.

Text:
{context}

Question: {question}
Answer:"""

    return prompt
