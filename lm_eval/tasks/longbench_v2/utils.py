"""Utility functions for LongBench v2 evaluation."""

import re
import string
from collections import Counter

import numpy as np


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if prediction exactly matches ground truth after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def qa_f1_score(prediction, ground_truth):
    """Compute QA F1 score."""
    return f1_score(prediction, ground_truth)


def qa_em_score(prediction, ground_truth):
    """Compute QA exact match score."""
    return exact_match_score(prediction, ground_truth)


def classification_score(prediction, ground_truth):
    """Score for classification tasks."""
    prediction = prediction.strip().lower()
    ground_truth = ground_truth.strip().lower()
    return 1.0 if prediction == ground_truth else 0.0


def retrieval_score(prediction, ground_truth):
    """Score for retrieval tasks."""
    # Extract numbers from prediction
    numbers = re.findall(r"\d+", prediction)
    if numbers:
        prediction_num = numbers[0]
    else:
        return 0.0

    return 1.0 if prediction_num == str(ground_truth) else 0.0


def count_score(prediction, ground_truth):
    """Score for counting tasks."""
    try:
        # Extract the first number from prediction
        numbers = re.findall(r"\d+", prediction)
        if not numbers:
            return 0.0
        pred_count = int(numbers[0])
        true_count = int(ground_truth)
        return 1.0 if pred_count == true_count else 0.0
    except (ValueError, TypeError):
        return 0.0


def code_similarity_score(prediction, ground_truth):
    """Score for code generation tasks using exact match of normalized code."""

    # Remove comments and normalize whitespace
    def normalize_code(code):
        # Remove single-line comments
        code = re.sub(r"//.*?\n", "\n", code)
        code = re.sub(r"#.*?\n", "\n", code)
        # Remove multi-line comments
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
        # Normalize whitespace
        code = " ".join(code.split())
        return code.strip()

    pred_normalized = normalize_code(prediction)
    truth_normalized = normalize_code(ground_truth)

    if pred_normalized == truth_normalized:
        return 1.0

    # Compute token-level F1 as partial credit
    pred_tokens = pred_normalized.split()
    truth_tokens = truth_normalized.split()
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(truth_tokens) if truth_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge_score(prediction, ground_truth):
    """Compute ROUGE-L score for summarization tasks."""

    def lcs(X, Y):
        m = len(X)
        n = len(Y)
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

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    lcs_length = lcs(prediction_tokens, ground_truth_tokens)

    precision = lcs_length / len(prediction_tokens) if prediction_tokens else 0
    recall = lcs_length / len(ground_truth_tokens) if ground_truth_tokens else 0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# Metric mapping for different task types
TASK_TO_METRIC = {
    # QA tasks
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "book_qa_eng": qa_f1_score,
    # Summarization tasks
    "gov_report": rouge_score,
    "multi_news": rouge_score,
    "book_sum": rouge_score,
    "samsum": rouge_score,
    # Classification tasks
    "trec": classification_score,
    # Retrieval tasks
    "passage_retrieval": retrieval_score,
    "kv_retrieval": retrieval_score,
    # Counting tasks
    "passage_count": count_score,
    # Code tasks
    "lcc": code_similarity_score,
    "repobench": code_similarity_score,
    "code_debug": code_similarity_score,
    # Other tasks
    "triviaqa": qa_f1_score,
    "paper_assistant": qa_f1_score,
}


def get_metric_for_task(task_name):
    """Get the appropriate metric function for a given task."""
    for key, metric in TASK_TO_METRIC.items():
        if key in task_name:
            return metric
    # Default to F1 score
    return qa_f1_score


def process_results_gen(doc, results):
    """Process generation results for LongBench v2 tasks."""
    completion = results[0]

    # Get the appropriate metric
    task_name = doc.get("task", "").lower()
    metric_fn = get_metric_for_task(task_name)

    # Handle multiple ground truth answers
    answers = doc.get("answers", doc.get("answer", ""))
    if not isinstance(answers, list):
        answers = [answers]

    # Compute score against all ground truths and take max
    scores = [metric_fn(completion, answer) for answer in answers]
    score = max(scores) if scores else 0.0

    return {"score": score}


def process_results_mc(doc, results):
    """Process multiple choice results for LongBench v2 tasks."""
    # For multiple choice, we expect log likelihoods
    gold_idx = doc.get("answer_idx", 0)

    # results should contain log likelihoods for each choice
    if isinstance(results, list) and len(results) > gold_idx:
        # Check if the highest likelihood corresponds to the correct answer
        pred_idx = np.argmax(results)
        return {"acc": 1.0 if pred_idx == gold_idx else 0.0}

    return {"acc": 0.0}
