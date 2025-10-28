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
from functools import partial
from typing import Any, Dict, List

import numpy as np

from lm_eval.tasks.score import utils
from lm_eval.tasks.score.utils import prompt_consistency_rate, robustness_doc_to_text


eval_logger = logging.getLogger(__name__)

TEMPLATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.json")

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "prompt_robustness"
OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY = "option_order_robustness"
NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY = "non_greedy_robustness"

QUESTION_KEY = "question"

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

mmlu_pro_prompt_consistency_rate = prompt_consistency_rate
mmlu_pro_robustness_doc_to_text = robustness_doc_to_text


prompt_robustness_process_docs = partial(
    utils.process_docs_add_prompts,
    templates_key=PROMPT_ROBUSTNESS_TEMPLATE_KEY,
    template_file_path=TEMPLATE_FILE_PATH,
)

option_order_robustness_process_docs = partial(
    utils.option_order_robustness_process_docs,
    template_file_path=TEMPLATE_FILE_PATH,
    templates_key=OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY,
    labels=LABELS,
)
non_greedy_robustness_process_docs = partial(
    utils.non_greedy_robustness_process_docs,
    template_file_path=TEMPLATE_FILE_PATH,
    templates_key=NON_GREEDY_ROBUSTNESS_TEMPLATE_KEY,
)


def non_greedy_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = utils.__postprocess_pred(results[0])
    final_answer = utils.translate_model_answer_to_labels(
        final_answer, option_format=doc["options_format"], labels=LABELS
    )
    question_id = doc["question_id"]
    category = doc["category"]
    gt = LABELS[doc["answer_index"]]

    return {"non_greedy_macro_accuracy": (question_id, final_answer, gt, category)}


def prompt_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = utils.__postprocess_pred(results[0])
    final_answer = utils.translate_model_answer_to_labels(
        final_answer, option_format=doc["options_format"], labels=LABELS
    )
    gt = LABELS[doc["answer_index"]]
    prompt_id = doc["prompt_id"]
    question_id = doc["question_id"]
    category = doc["category"]
    return {
        f"{prompt_id}_macro_accuracy": (
            question_id,
            prompt_id,
            final_answer,
            gt,
            category,
        ),
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
    category = doc["category"]
    return {
        f"per_option_macro_accuracy_{always_same_option}": (
            question_id,
            always_same_option,
            final_answer,
            gt,
            category,
        ),
        "options_consistency_rate": (
            question_id,
            always_same_option,
            final_answer,
            original_answer_index,
            answer_index,
        ),
    }


def per_prompt_macro_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = {}
    for result in results:
        question_id, prompt_id, final_answer, gt, category = result
        if prompt_id != p_id:
            continue
        if category not in accuracies:
            accuracies[category] = []
        accuracies[category].append(final_answer == gt)

    for key in accuracies:
        accuracies[key] = sum(accuracies[key]) / len(accuracies[key])
        eval_logger.info(
            f"Prompt - {prompt_id}, category - {key} accuracy: {accuracies[key]}"
        )

    return np.round(np.mean([v for v in accuracies.values()]), 4)


per_prompt_accuracy_0 = partial(per_prompt_macro_accuracy, p_id=0)
per_prompt_accuracy_1 = partial(per_prompt_macro_accuracy, p_id=1)
per_prompt_accuracy_2 = partial(per_prompt_macro_accuracy, p_id=2)
per_prompt_accuracy_3 = partial(per_prompt_macro_accuracy, p_id=3)
per_prompt_accuracy_4 = partial(per_prompt_macro_accuracy, p_id=4)
per_prompt_accuracy_5 = partial(per_prompt_macro_accuracy, p_id=5)
per_prompt_accuracy_6 = partial(per_prompt_macro_accuracy, p_id=6)
per_prompt_accuracy_7 = partial(per_prompt_macro_accuracy, p_id=7)
per_prompt_accuracy_8 = partial(per_prompt_macro_accuracy, p_id=8)
per_prompt_accuracy_9 = partial(per_prompt_macro_accuracy, p_id=9)


def per_option_macro_accuracy(results: List[Dict[str, Any]], always_opt="a") -> float:
    accuracies = {}
    for result in results:
        question_id, always_same_option, final_answer, gt, category = result
        if always_opt != always_same_option:
            continue
        if category not in accuracies:
            accuracies[category] = []
        accuracies[category].append(int(final_answer == gt))

    for key in accuracies:
        accuracies[key] = sum(accuracies[key]) / len(accuracies[key])
        eval_logger.info(
            f"Prompt - {always_opt.upper()}, category - {key} accuracy: {accuracies[key]}"
        )

    return np.round(np.mean([v for v in accuracies.values()]), 4)


per_option_macro_accuracy_a = partial(per_option_macro_accuracy, always_opt="A")
per_option_macro_accuracy_b = partial(per_option_macro_accuracy, always_opt="B")
per_option_macro_accuracy_c = partial(per_option_macro_accuracy, always_opt="C")
per_option_macro_accuracy_d = partial(per_option_macro_accuracy, always_opt="D")
per_option_macro_accuracy_e = partial(per_option_macro_accuracy, always_opt="E")
per_option_macro_accuracy_f = partial(per_option_macro_accuracy, always_opt="F")
per_option_macro_accuracy_g = partial(per_option_macro_accuracy, always_opt="G")
per_option_macro_accuracy_h = partial(per_option_macro_accuracy, always_opt="H")
per_option_macro_accuracy_i = partial(per_option_macro_accuracy, always_opt="I")
per_option_macro_accuracy_j = partial(per_option_macro_accuracy, always_opt="J")

options_consistency_rate = partial(utils.options_consistency_rate, labels=LABELS)


def non_greedy_macro_accuracy(results: List[Dict[str, Any]]) -> float:
    accuracies = {}
    for result in results:
        question_id, final_answer, gt, category = result
        if category not in accuracies:
            accuracies[category] = []
        accuracies[category].append(final_answer == gt)

    for key in accuracies:
        accuracies[key] = sum(accuracies[key]) / len(accuracies[key])
        eval_logger.info(f"Non greedy, category - {key} accuracy: {accuracies[key]}")

    return np.round(np.mean([v for v in accuracies.values()]), 4)
