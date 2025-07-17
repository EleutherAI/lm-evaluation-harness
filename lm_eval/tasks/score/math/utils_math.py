# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from functools import partial
from itertools import combinations
from typing import Any, Dict, List

import datasets
import numpy as np

from lm_eval.tasks.score import utils
from lm_eval.tasks.score.math.math_grader import (
    extract_answer,
    math_equal,
    normalize_answer_string,
)
from lm_eval.tasks.score.utils import robustness_doc_to_text


eval_logger = logging.getLogger(__name__)

TEMPLATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.json")

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "prompt_robustness"
NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY = "non_greedy_robustness"

math_robustness_doc_to_text = robustness_doc_to_text


def find_boxed_entries(answer_str):
    stack = []
    results = []
    i = 0

    while i < len(answer_str):
        if answer_str[i : i + 7] == "\\boxed{":
            stack.append(i + 7)
            i += 7
        elif answer_str[i] == "{":
            if stack:
                stack.append(i + 1)
            i += 1
        elif answer_str[i] == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    results.append(answer_str[start:i])
            i += 1
        else:
            i += 1

    if len(results) == 0:
        raise ValueError("Not enough boxed entries")
    else:
        results = [normalize_answer_string(result) for result in results]

    if len(results) == 1:
        # Single boxed entry, trivial case
        return results

    else:
        # Multiple boxed entries. There are two cases possible
        # (a) The reference solution has the same question answered in multiple ways
        # (b) The answer is split across multiple boxed entries and we need to merge
        result_equal = True
        for idx in range(len(results) - 1):
            if not (results[idx] == results[idx + 1]):
                result_equal = False
                break

        if result_equal:
            # Same problem solved in multiple ways
            return [results[0]]
        else:
            return results


def extract_answer_dataset(solution: str, problem: str, corrected_answers: list) -> str:
    entries = find_boxed_entries(solution)

    if len(entries) == 1:
        parsed_answer = entries[0]

    if len(entries) > 1:
        for item in corrected_answers:
            if item["problem"] == problem:
                parsed_answer = item["answer"]
                break
        else:
            parsed_answer = ", ".join(entries)

    if not (
        ("Find the equation" in problem)
        or ("Enter the equation" in problem)
        or ("What is the equation" in problem)
        or ("described by the equation" in problem)
        or ("Find an equation" in problem)
    ) and ("=" in parsed_answer):
        if parsed_answer.count("=") == 1:
            # For greater count, it means we're just predicting values of multiple variables
            parsed_answer = parsed_answer.split("=")[1]
    return parsed_answer


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict, idx, corrected_answer) -> dict:
        out_doc = {
            "question": doc["problem"],
            "question_id": idx,
            "solution": doc["solution"],
            "answer": extract_answer_dataset(
                doc["solution"], doc["problem"], corrected_answer
            ),
        }
        return out_doc

    corrected_answer_path = os.path.join(
        os.path.dirname(__file__), "to_be_fixed_questions.json"
    )

    with open(corrected_answer_path, "r") as f:
        corrected_answers = json.load(f)

    return dataset.map(
        partial(_process_doc, corrected_answer=corrected_answers), with_indices=True
    )


def prompt_robustness_process_docs(doc: datasets.Dataset) -> datasets.Dataset:
    doc = process_docs(doc)
    return utils.process_docs_add_prompts(
        doc,
        templates_key=PROMPT_ROBUSTNESS_TEMPLATE_KEY,
        template_file_path=TEMPLATE_FILE_PATH,
    )


def non_greedy_robustness_process_docs(doc: datasets.Dataset) -> datasets.Dataset:
    doc = process_docs(doc)
    return utils.non_greedy_robustness_process_docs(
        doc,
        templates_key=NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY,
        template_file_path=TEMPLATE_FILE_PATH,
    )


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    answer = extract_answer(results[0])

    if math_equal(answer, doc["answer"]):
        retval = 1
    else:
        retval = 0

    prompt_id = doc["prompt_id"]

    results = {
        f"{prompt_id}_accuracy": (prompt_id, retval),
        "consistency_rate": (doc["question_id"], answer),
    }
    return results


def non_greedy_robustness_process_results(
    doc: dict, results: List[str]
) -> Dict[str, int]:
    answer = extract_answer(results[0])
    return {"non_greedy_accuracy": (doc["question_id"], answer, doc["answer"], None)}


def per_prompt_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = []
    for result in results:
        prompt_id, retval = result
        if prompt_id != p_id:
            continue
        accuracies.append(retval)

    accuracy = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {prompt_id} accuracy: {accuracy}")

    return np.round(accuracy, 4)


per_prompt_accuracy_0 = partial(per_prompt_accuracy, p_id=0)
per_prompt_accuracy_1 = partial(per_prompt_accuracy, p_id=1)
per_prompt_accuracy_2 = partial(per_prompt_accuracy, p_id=2)
per_prompt_accuracy_3 = partial(per_prompt_accuracy, p_id=3)
per_prompt_accuracy_4 = partial(per_prompt_accuracy, p_id=4)
per_prompt_accuracy_5 = partial(per_prompt_accuracy, p_id=5)
per_prompt_accuracy_6 = partial(per_prompt_accuracy, p_id=6)
per_prompt_accuracy_7 = partial(per_prompt_accuracy, p_id=7)
per_prompt_accuracy_8 = partial(per_prompt_accuracy, p_id=8)
per_prompt_accuracy_9 = partial(per_prompt_accuracy, p_id=9)


def calculate_consistency_rate(responses: List[List[str]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    total_similarity = 0
    total_combinations = 0

    for response_set in responses:
        pairs = combinations(response_set, 2)
        num_pairs = len(response_set) * (len(response_set) - 1) / 2
        total_combinations += num_pairs
        for answer1, answer2 in pairs:
            total_similarity += int(math_equal(answer1, answer2))

    return total_similarity / total_combinations if total_combinations > 0 else 0.0


def math_prompt_consistency_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    question_answers_dict = {}

    for result in results:
        question_id, answer = result
        if question_id not in question_answers_dict:
            question_answers_dict[question_id] = []
        question_answers_dict[question_id].append(answer)

    question_answers_list = [answers for answers in question_answers_dict.values()]

    return calculate_consistency_rate(question_answers_list)


def non_greedy_accuracy(results: List[Dict[str, Any]]) -> float:
    accuracies = []
    for result in results:
        question_id, final_answer, gt, _ = result
        if math_equal(final_answer, gt):
            retval = 1
        else:
            retval = 0
        accuracies.append(retval)

    accuracy = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Non greedy accuracy: {accuracy}")

    return np.round(accuracy, 4)
