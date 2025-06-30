# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
import re
import string
import sys
from functools import partial
from itertools import combinations
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset


eval_logger = logging.getLogger(__name__)


NUMERALS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
ROMAN_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def __repeat_elements(lst, n):
    result = []
    for element in lst:
        result.extend([element] * n)
    return result


def process_docs_add_prompts(
    doc: Dataset,
    templates_key: str,
    template_file_path: str,
    dataset_specific_preprocess: callable = None,
) -> Dataset:
    try:
        with open(template_file_path) as f:
            prompt_templates = json.load(f)[templates_key]
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()
    if dataset_specific_preprocess is not None:
        doc = dataset_specific_preprocess(doc)

    def process_batch(batch):
        n = len(prompt_templates)
        initial_len = len(next(iter(batch.values())))

        result = {key: __repeat_elements(values, n) for key, values in batch.items()}
        result["prompt_id"] = list(range(n)) * initial_len
        result["prompt"] = [prompt_templates[i]["prompt"] for i in result["prompt_id"]]
        if "options_format" in prompt_templates[0]:
            result["options_format"] = [
                prompt_templates[i]["options_format"] for i in result["prompt_id"]
            ]
        return result

    return doc.map(process_batch, batched=True)


def option_order_robustness_process_docs(
    doc: Dataset,
    template_file_path: str,
    templates_key: str,
    labels: list,
    dataset_specific_preprocess: callable = None,
) -> Dataset:
    try:
        with open(template_file_path) as f:
            prompt_template = json.load(f)[templates_key]
            prompt = prompt_template["prompt"]
            options_format = prompt_template["options_format"]
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()

    if dataset_specific_preprocess is not None:
        doc = dataset_specific_preprocess(doc)

    def repeat_doc_swap_correct_answer(batched_docs):
        initial_len = len(next(iter(batched_docs.values())))
        keys = list(batched_docs.keys())
        new_batched_docs = {key: [] for key in keys}
        new_batched_docs["always_same_option"] = []
        new_batched_docs["prompt"] = []
        new_batched_docs["options_format"] = []
        new_batched_docs["original_answer_index"] = []

        for doc_ind in range(initial_len):
            for label_ind, label in enumerate(labels):
                new_batched_docs["original_answer_index"].append(
                    batched_docs["answer_index"][doc_ind]
                )
                for key in keys:
                    new_batched_docs[key].append(
                        copy.deepcopy(batched_docs[key][doc_ind])
                    )
                    if label_ind < len(batched_docs["options"][doc_ind]):
                        if key == "options":
                            # Swap correct answer with label_ind option
                            new_batched_docs[key][-1][label_ind] = batched_docs[
                                "options"
                            ][doc_ind][batched_docs["answer_index"][doc_ind]]
                            new_batched_docs[key][-1][
                                batched_docs["answer_index"][doc_ind]
                            ] = batched_docs["options"][doc_ind][label_ind]

                        if key == "answer_index":
                            new_batched_docs[key][-1] = label_ind

                        if key == "answer":
                            new_batched_docs[key][-1] = label

                new_batched_docs["always_same_option"].append(label)
                new_batched_docs["prompt"].append(prompt)
                new_batched_docs["options_format"].append(options_format)
        return new_batched_docs

    return doc.map(repeat_doc_swap_correct_answer, batched=True)


def non_greedy_robustness_process_docs(
    doc: Dataset,
    templates_key: str,
    template_file_path: str,
    dataset_specific_preprocess: callable = None,
) -> Dataset:
    try:
        with open(template_file_path) as f:
            prompt_template = json.load(f)[templates_key]
            prompt = prompt_template["prompt"]
            options_format = prompt_template.get("options_format", None)
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()

    if dataset_specific_preprocess is not None:
        doc = dataset_specific_preprocess(doc)

    def add_prompt_col(batched_docs):
        initial_len = len(next(iter(batched_docs.values())))
        new_batched_docs = copy.deepcopy(batched_docs)
        new_batched_docs["prompt"] = [prompt] * initial_len
        if options_format is not None:
            new_batched_docs["options_format"] = [options_format] * initial_len

        return new_batched_docs

    return doc.map(add_prompt_col, batched=True)


def robustness_doc_to_text(doc: Dataset) -> str:
    upper_case = string.ascii_uppercase
    lower_case = string.ascii_lowercase
    prompt = doc["prompt"]
    options_format = doc.get("options_format", "")
    question = doc["question"]
    catrgory = doc.get("category", "")
    options = None
    if options_format:
        options = "".join(
            [
                options_format.format(
                    letter=upper_case[i],
                    option=doc["options"][i],
                    numeral=NUMERALS[i],
                    roman_numeral=ROMAN_NUMERALS[i],
                    lower_case_letter=lower_case[i],
                )
                for i in range(len(doc["options"]))
            ]
        )
    return prompt.format(question=question, options=options, category=catrgory)


def __postprocess_pred(pred):
    if "the best answer is" not in pred.lower():
        return pred
    pred_proc = (
        pred.lower().split("the best answer is ")[-1].split("\n")[0].split(" ")[0]
    )
    pred_proc = re.sub(r"[^a-zA-Z0-9]", "", pred_proc).strip()
    return pred_proc.upper()


def translate_model_answer_to_labels(answer, labels, option_format=None):
    answer = answer.upper()

    if option_format is None:
        return answer

    elif "numeral" in option_format:
        if "roman" in option_format:
            if answer not in ROMAN_NUMERALS:
                return answer
            else:
                return labels[ROMAN_NUMERALS.index(answer)]

        if answer not in NUMERALS:
            return answer
        else:
            return labels[NUMERALS.index(answer)]

    return answer


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
            total_similarity += int(answer1 == answer2)

    return total_similarity / total_combinations if total_combinations > 0 else 0.0


def prompt_consistency_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    question_answers_dict = {}

    for result in results:
        question_id, prompt_id, final_answer, gt = result
        if question_id not in question_answers_dict:
            question_answers_dict[question_id] = []
        question_answers_dict[question_id].append(final_answer)

    question_answers_list = [answers for answers in question_answers_dict.values()]

    return calculate_consistency_rate(question_answers_list)


def options_consistency_rate(results: List[Dict[str, Any]], labels) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    question_answers_dict = {}
    for result in results:
        (
            question_id,
            always_same_option,
            final_answer,
            original_answer_index,
            answer_index,
        ) = result
        if final_answer == labels[original_answer_index]:
            final_answer = always_same_option
        if final_answer == always_same_option:
            final_answer = labels[original_answer_index]
        if question_id not in question_answers_dict:
            question_answers_dict[question_id] = []
        question_answers_dict[question_id].append(final_answer)

    question_answers_list = [answers for answers in question_answers_dict.values()]

    return calculate_consistency_rate(question_answers_list)
