import copy
from functools import partial
from itertools import combinations
import json
import os
import string
import sys
from datasets import Dataset
from typing import Any, Dict, List
from lm_eval.utils import eval_logger
import re
import numpy as np

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "v0.4_templates"
OPTION_FORMAT_ROBUSTNESS_TEMPLATE_KEY = "options_format_robustness"
OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY = "same_options_prompt"


def __repeat_elements(lst, n):
    result = []
    for element in lst:
        result.extend([element] * n)
    return result

def process_docs_add_prompts(doc: Dataset, templates_key) -> Dataset:
    def process_batch(batch):
        try:
            template_file = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
            with open(template_file) as f:
                prompt_templates = json.load(f)[templates_key] #todo
        except FileNotFoundError:
            eval_logger.error("Prompt templates not found")
            sys.exit()

        n = len(prompt_templates)
        initial_len = len(next(iter(batch.values())))

        result = {key: __repeat_elements(values, n) for key, values in batch.items()}
        result["prompt_id"] = list(range(n)) * initial_len
        result["prompt"] = [prompt_templates[i]["prompt"] for i in result["prompt_id"]]
        result["options_format"] = [prompt_templates[i]["options_format"] for i in result["prompt_id"]]
        return result
    return doc.map(process_batch, batched=True)

prompt_robustness_process_docs = partial(process_docs_add_prompts, templates_key=PROMPT_ROBUSTNESS_TEMPLATE_KEY)
option_format_robustness_process_docs = partial(process_docs_add_prompts, templates_key=OPTION_FORMAT_ROBUSTNESS_TEMPLATE_KEY)

def option_order_robustness_process_docs(doc: Dataset) -> Dataset:
    try:
        template_file = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
        with open(template_file) as f:
            prompt_template = json.load(f)[OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY] #todo
            prompt = prompt_template["prompt"]
            options_format = prompt_template["options_format"]
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()


    def repeat_doc_swap_correct_answer(batched_docs):

        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        initial_len = len(next(iter(batched_docs.values())))
        keys = list(batched_docs.keys())
        new_batched_docs = {key: [] for key in keys}
        new_batched_docs["always_same_option"] = []
        new_batched_docs["prompt"] = []
        new_batched_docs["options_format"] = []
        new_batched_docs["original_answer_index"] = []
        
        for doc_ind in range(initial_len):
            for label_ind, label in enumerate(labels):
                new_batched_docs["original_answer_index"].append(batched_docs["answer_index"][doc_ind])
                for key in keys:
                    new_batched_docs[key].append(copy.deepcopy(batched_docs[key][doc_ind]))
                    if label_ind < len(batched_docs["options"][doc_ind]):
                        if key == "options":
                            # Swap correct answer with label_ind option
                            new_batched_docs[key][-1][label_ind] = batched_docs["options"][doc_ind][batched_docs["answer_index"][doc_ind]]
                            new_batched_docs[key][-1][batched_docs["answer_index"][doc_ind]] = batched_docs["options"][doc_ind][label_ind]
                        
                        if key == "answer_index":
                            new_batched_docs[key][-1] = label_ind

                        if key == "answer":
                            new_batched_docs[key][-1] = label

                new_batched_docs["always_same_option"].append(label)
                new_batched_docs["prompt"].append(prompt)
                new_batched_docs["options_format"].append(options_format)
        return new_batched_docs
    return doc.map(repeat_doc_swap_correct_answer, batched=True)



def robustness_doc_to_text(doc: Dataset) -> str:
    _l = string.ascii_uppercase
    numerals = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    prompt = doc["prompt"]
    options_format = doc["options_format"]
    options = "".join([options_format.format(letter=_l[i], option=doc['options'][i], numeral=numerals[i]) for i in range(len(doc["options"]))])
    return prompt.format(question=doc["question"], options=options, category=doc["category"])

def __postprocess_pred(pred):
    if "the best answer is" not in pred.lower():
        return pred
    pred_proc = pred.lower().split("the best answer is ")[-1].split(' ')[0]
    pred_proc = re.sub(r"[^a-zA-Z]", "", pred_proc).strip().upper()
    return pred_proc

def prompt_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0]).lower()
    gt = doc["answer"].lower()
    prompt_id = doc["prompt_id"]
    question_id = doc["question_id"]
    category = doc["category"]
    return {
        f"prompt_{prompt_id}_macro_accuracy": (question_id, prompt_id, final_answer, gt, category),
        "per_prompt_accuracy_std": (question_id, prompt_id, final_answer, gt, category),
        "prompt_consistency_rate": (question_id, prompt_id, final_answer, gt)}

def option_order_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0]).lower()
    gt = doc["answer"].lower()
    always_same_option = doc["always_same_option"]
    question_id = doc["question_id"]
    original_answer_index = doc["original_answer_index"]
    answer_index = doc["answer_index"]
    
    return {"per_option_accuracy_std": (question_id, always_same_option, final_answer, gt),
             "options_consistency_rate": (question_id, always_same_option, final_answer, original_answer_index, answer_index)}

def per_prompt_macro_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = {}
    for result in results:
        question_id, prompt_id, final_answer, gt, category = result
        if prompt_id != p_id:
            continue
        if category not in accuracies:
            accuracies[category] = []
        accuracies[category].append(int(final_answer == gt))
    
    for key in accuracies:
        accuracies[key] = sum(accuracies[key]) / len(accuracies[key])
        eval_logger.info(f"Prompt - {prompt_id}, category - {key} accuracy: {accuracies[key]}")
    
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

def per_prompt_accuracy_std(results: List[Dict[str, Any]]) -> float:
    """
    Computes std of the per_prompt_accuracy.
    Input: List of dictionaries, each containing the keys "question_id", "prompt_id", "final_answer", "gt", "category"
    Output: accuracy std through prompt_id's
    """
    prompt_accuracies = {}
    for result in results:
        question_id, prompt_id, final_answer, gt, category = result
        if f"accuracy_{prompt_id}" not in prompt_accuracies:
            prompt_accuracies[f"accuracy_{prompt_id}"] = {category:[]}
        if category not in prompt_accuracies[f"accuracy_{prompt_id}"]:
            prompt_accuracies[f"accuracy_{prompt_id}"][category] = []
        prompt_accuracies[f"accuracy_{prompt_id}"][category].append(int(final_answer == gt))

    for key in prompt_accuracies:
        per_prompt_accuracies = []
        for category in prompt_accuracies[key]:
            per_prompt_accuracies.append(sum(prompt_accuracies[key][category]) / len(prompt_accuracies[key][category]))
        prompt_accuracies[key] = np.mean(per_prompt_accuracies)
        eval_logger.info(f"Prompt {key} accuracy: {prompt_accuracies[key]}")
    
    std_dev = np.std([v for v in prompt_accuracies.values()])
    
    return np.round(std_dev, 4)


def per_option_accuracy_std(results: List[Dict[str, Any]]) -> float:
    """
    Computes the mean accuracy of the per_option_accuracy_std metric.
    Input: List of dictionaries, each containing the keys "question_id", "always_same_option", "final_answer", "gt"
    Output: Dictionary containing accuracy_`option` keys with the mean accuracy for each option
    """
    options_accuracies = {}
    for result in results:
        question_id, always_same_option, final_answer, gt = result
        if f"accuracy_{always_same_option}" not in options_accuracies:
            options_accuracies[f"accuracy_{always_same_option}"] = []
        options_accuracies[f"accuracy_{always_same_option}"].append(int(final_answer == gt))

    for key in options_accuracies:
        options_accuracies[key] = sum(options_accuracies[key]) / len(options_accuracies[key])
        eval_logger.info(f"Option {key} accuracy: {options_accuracies[key]}")
    
    std_dev = np.std([v for v in options_accuracies.values()])
    
    return np.round(std_dev,3)


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
            total_similarity +=int(answer1==answer2)

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


def options_consistency_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate the Consistency Rate (CR) for a given set of responses.

    Args:
    responses: List of lists, where each inner list contains responses to the same question.

    Returns:
    The consistency rate as a float.
    """
    question_answers_dict = {}
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for result in results:
        question_id, always_same_option, final_answer, original_answer_index, answer_index = result
        if final_answer == labels[original_answer_index]:
            final_answer = always_same_option
        if final_answer == always_same_option:
            final_answer = labels[original_answer_index]
        if question_id not in question_answers_dict:
            question_answers_dict[question_id] = []
        question_answers_dict[question_id].append(final_answer)


    question_answers_list = [answers for answers in question_answers_dict.values()]

    return calculate_consistency_rate(question_answers_list)
