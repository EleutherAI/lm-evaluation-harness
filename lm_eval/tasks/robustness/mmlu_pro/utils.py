import copy
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

PROMPT_TEMPLATES_VERSION_02 = "v0.3_templates"
PROMPT_TEMPLATES_SAME_OPTIONS = "same_options_prompt"


def __repeat_elements(lst, n):
    result = []
    for element in lst:
        result.extend([element] * n)
    return result

def prompt_robustness_process_docs(doc: Dataset) -> Dataset:
    def process_batch(batch):
        try:
            template_file = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
            with open(template_file) as f:
                prompt_templates = json.load(f)[PROMPT_TEMPLATES_VERSION_02] #todo
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



def options_robustness_process_docs(doc: Dataset) -> Dataset:
    try:
        template_file = os.path.join(os.path.dirname(__file__), "prompt_templates.json")
        with open(template_file) as f:
            prompt_template = json.load(f)[PROMPT_TEMPLATES_SAME_OPTIONS] #todo
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
    options = "\n".join([f"{_l[i]}) {doc['options'][i]}" for i in range(len(doc["options"]))])

    prompt = doc["prompt"]
    options_format = doc["options_format"]
    options = "".join([options_format.format(letter=_l[i], option=doc['options'][i]) for i in range(len(doc["options"]))])
    return prompt.format(question=doc["question"], options=options, category=doc["category"])

def __postprocess_pred(predict_str):
        """
        Todo: make code simpler
        """        
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        separators = ["<extra_id_1>", "<|eot_id|>", "\n", " ", "\\n"]

        for l in labels:
            if predict_str.startswith(f"{l}:"):
                return l
            if predict_str.startswith(f"{l}\n"):
                return l
            if f" {l}: " in predict_str:
                return l

        if 'which corresponds to' in predict_str:
            predict_str = predict_str.split('which corresponds to')[-1].strip()
            for word in predict_str.split():
                only_letters = re.sub(r"[^a-zA-Z]", "", word).strip()
                for choice in ["A", "B", "C", "D", "E"]:
                    if only_letters == choice:
                        return only_letters

        predict_str = predict_str.strip()
        to_split = ['the correct answer is', 'the correct response is', 'the correct option is', 'the most accurate answer is',
                    'the best answer here is', 'the best answer is', 'the answer must be', 'the most accurate response is',
                    'the most fitting response is', 'the most fitting answer is', 'the answer is', 'answer: ' ]
        for answer_str in to_split:
            if answer_str in predict_str.lower():
                predict_str = predict_str.lower().split(answer_str)
                predict_str = [i for i in predict_str if len(i) > 0]
                if len(predict_str) > 0:
                    predict_str = predict_str[-1].upper()
                else:
                    return "EMPTY"
                predict_str = predict_str.strip(string.punctuation).strip()

        for separator in separators:
            if separator in predict_str:
                predict_str = predict_str.split(separator)[0].strip()

        delimiters = [" ", ",", "."]
        quotes = ["'", '"', "'", "`", "`", "."]
        # if LABEL_TO_ID[task_name] doesn't contain any quotes, remove them from predict_str
        if not any([quote in "".join(labels) for quote in quotes]):
            for quote in quotes:
                predict_str = predict_str.replace(quote, "")

        # remove repeated labels while making sure only the label is repeated
        for label in labels:
            label_count = predict_str.count(label)
            if label_count > 1:
                for delimiter in delimiters:
                    if delimiter in predict_str:
                        repeated_label = delimiter.join([label] * label_count)
                        if repeated_label == predict_str:
                            predict_str = predict_str.split(delimiter)[0]
                            break
        replace_dict = {',':'', '\\_': '_', '.':'', ':':''}
        for key, value in replace_dict.items():
            predict_str = predict_str.replace(key, value)

        predict_str = predict_str.strip(string.punctuation).strip()
        return predict_str

def prompt_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0]).lower()
    gt = doc["answer"].lower()
    prompt_id = doc["prompt_id"]
    question_id = doc["question_id"]
    return {"per_prompt_accuracy_std": (question_id, prompt_id, final_answer, gt),
            "prompt_consistency_rate": (question_id, prompt_id, final_answer, gt)}

def options_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0]).lower()
    gt = doc["answer"].lower()
    always_same_option = doc["always_same_option"]
    question_id = doc["question_id"]
    original_answer_index = doc["original_answer_index"]
    answer_index = doc["answer_index"]
    
    return {"per_option_accuracy_std": (question_id, always_same_option, final_answer, gt),
             "options_consistency_rate": (question_id, always_same_option, final_answer, original_answer_index, answer_index)}


def per_prompt_accuracy_std(results: List[Dict[str, Any]]) -> float:
    """
    Computes the mean accuracy of the per_prompt_accuracy_std metric.
    Input: List of dictionaries, each containing the keys "question_id", "prompt_id", "final_answer", "gt"
    Output: Dictionary containing accuracy_`prompt_id` keys with the mean accuracy for each prompt_id
    """
    prompt_accuracies = {}
    for result in results:
        question_id, prompt_id, final_answer, gt = result
        if f"accuracy_{prompt_id}" not in prompt_accuracies:
            prompt_accuracies[f"accuracy_{prompt_id}"] = []
        prompt_accuracies[f"accuracy_{prompt_id}"].append(int(final_answer == gt))

    for key in prompt_accuracies:
        prompt_accuracies[key] = sum(prompt_accuracies[key]) / len(prompt_accuracies[key])
        eval_logger.info(f"Prompt {key} accuracy: {prompt_accuracies[key]}")
    
    std_dev = np.std([v for v in prompt_accuracies.values()])
    
    return np.round(std_dev,3)


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
