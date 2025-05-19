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

import logging
import os
import re
from functools import partial
from typing import Any, Dict, List

import numpy as np
from datasets import Dataset

from lm_eval.tasks.score import utils
from lm_eval.tasks.score.utils import prompt_consistency_rate, robustness_doc_to_text


eval_logger = logging.getLogger(__name__)

TEMPLATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.json")

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "prompt_robustness"
OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY = "option_order_robustness"
NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY = "non_greedy_robustness"

QUESTION_KEY = "query"
ANSWER_INDEX_KEY = "gold"
OPTIONS_KEY = "choices"

LABELS = ["A", "B", "C", "D", "E"]

agi_eval_prompt_consistency_rate = prompt_consistency_rate
agi_eval_robustness_doc_to_text = robustness_doc_to_text


def initial_process_docs(doc: Dataset) -> Dataset:
    """
    add question_id to the documents
    """

    bracket_pattern = r"^\([A-E]\)"
    letter_space = r"^[A-E] "
    letter_question_space = r"^[A-E]\? "

    def __process(_doc, idx):
        if "question" not in _doc:
            question = _doc[QUESTION_KEY].split(" Answer Choices:")[0]
            if question.startswith("Q: "):
                question = question[3:]
            _doc["question"] = question
        if "question_id" not in _doc:
            _doc["question_id"] = idx
        if "answer_index" not in _doc:
            _doc["answer_index"] = _doc[ANSWER_INDEX_KEY][0]
        if "answer" not in _doc:
            _doc["answer"] = LABELS[_doc["answer_index"]]
        if "options" not in _doc:
            prepared_options = []
            for option in _doc[OPTIONS_KEY]:
                if re.match(bracket_pattern, option):
                    prepared_options.append(option[3:])
                elif re.match(letter_space, option):
                    prepared_options.append(option[2:])
                elif re.match(letter_question_space, option):
                    prepared_options.append(option[3:])
                else:
                    prepared_options.append(option)
            _doc["options"] = prepared_options
        return _doc

    return doc.map(__process, with_indices=True)


prompt_robustness_process_docs = partial(
    utils.process_docs_add_prompts,
    templates_key=PROMPT_ROBUSTNESS_TEMPLATE_KEY,
    template_file_path=TEMPLATE_FILE_PATH,
    dataset_specific_preprocess=initial_process_docs,
)

option_order_robustness_process_docs = partial(
    utils.option_order_robustness_process_docs,
    template_file_path=TEMPLATE_FILE_PATH,
    templates_key=OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY,
    labels=LABELS[:-1],
    dataset_specific_preprocess=initial_process_docs,
)

non_greedy_robustness_process_docs = partial(
    utils.non_greedy_robustness_process_docs,
    templates_key=NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY,
    template_file_path=TEMPLATE_FILE_PATH,
    dataset_specific_preprocess=initial_process_docs,
)


def prompt_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = utils.__postprocess_pred(results[0])
    final_answer = utils.translate_model_answer_to_labels(
        final_answer, option_format=doc["options_format"], labels=LABELS
    )
    gt = LABELS[doc["answer_index"]]
    prompt_id = doc["prompt_id"]
    question_id = doc["question_id"]
    return {
        f"{prompt_id}_accuracy": (question_id, prompt_id, final_answer, gt),
        "consistency_rate": (question_id, prompt_id, final_answer, gt),
    }


def option_order_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = utils.__postprocess_pred(results[0])
    final_answer = utils.translate_model_answer_to_labels(
        final_answer, option_format=doc["options_format"], labels=LABELS
    )
    gt = LABELS[doc["answer_index"]]
    always_same_option = doc["always_same_option"]
    question_id = doc["question_id"]
    original_answer_index = doc["original_answer_index"]
    answer_index = (doc["answer_index"],)
    return {
        f"per_option_accuracy_{always_same_option}": (
            question_id,
            always_same_option,
            final_answer,
            gt,
        ),
        "options_consistency_rate": (
            question_id,
            always_same_option,
            final_answer,
            original_answer_index,
            answer_index,
        ),
    }


def non_greedy_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = utils.__postprocess_pred(results[0])
    final_answer = utils.translate_model_answer_to_labels(
        final_answer, option_format=doc["options_format"], labels=LABELS
    )
    question_id = doc["question_id"]
    gt = LABELS[doc["answer_index"]]

    return {"non_greedy_accuracy": (question_id, final_answer, gt, None)}


def per_prompt_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = []
    for result in results:
        question_id, prompt_id, final_answer, gt = result
        if prompt_id != p_id:
            continue
        accuracies.append(final_answer == gt)

    accuracie = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {prompt_id} accuracy: {accuracie}")

    return np.round(accuracie, 4)


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


def per_option_accuracy(results: List[Dict[str, Any]], always_opt="a") -> float:
    accuracies = []
    for result in results:
        question_id, always_same_option, final_answer, gt = result
        if always_opt != always_same_option:
            continue
        accuracies.append(int(final_answer == gt))

    accuracie = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {always_opt.upper()} accuracy: {accuracie}")

    return np.round(accuracie, 4)


per_option_accuracy_a = partial(per_option_accuracy, always_opt="A")
per_option_accuracy_b = partial(per_option_accuracy, always_opt="B")
per_option_accuracy_c = partial(per_option_accuracy, always_opt="C")
per_option_accuracy_d = partial(per_option_accuracy, always_opt="D")

options_consistency_rate = partial(utils.options_consistency_rate, labels=LABELS)


def non_greedy_accuracy(results: List[Dict[str, Any]]) -> float:
    accuracies = []
    for result in results:
        question_id, final_answer, gt, category = result

        accuracies.append(final_answer == gt)

    accuracy = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Non greedy accuracy: {accuracy}")

    return np.round(accuracy, 4)
