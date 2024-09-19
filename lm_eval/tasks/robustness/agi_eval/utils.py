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

TEMPLATE_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompt_templates.json")

PROMPT_ROBUSTNESS_TEMPLATE_KEY = "prompt_robustness"
OPTION_FORMAT_ROBUSTNESS_TEMPLATE_KEY = "option_format_robustness"
OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY = "option_order_robustness"

QUESTION_KEY = "query"
ANSWER_INDEX_KEY = "gold"
OPTIONS_KEY = "choices"

LABELS = ['A', 'B', 'C', 'D', 'E']

def __repeat_elements(lst, n):
    result = []
    for element in lst:
        result.extend([element] * n)
    return result

def initial_process_docs(doc: Dataset) -> Dataset:
    """
    add question_id to the documents
    """

    bracket_pattern = r'^\([A-E]\)'
    letter_space = r'^[A-E] '
    letter_question_space = r'^[A-E]\? '
    
    def __process(_doc, idx):
        if "question" not in _doc:
            _doc["question"] = _doc[QUESTION_KEY].split(" Answer Choices:")[0]
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

def process_docs_add_prompts(doc: Dataset, templates_key) -> Dataset:
    doc = initial_process_docs(doc)

    try:
        with open(TEMPLATE_FILE_PATH) as f:
            prompt_templates = json.load(f)[templates_key] #todo
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()

    def process_batch(batch):

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
    doc = initial_process_docs(doc)

    try:
        with open(TEMPLATE_FILE_PATH) as f:
            prompt_template = json.load(f)[OPTION_ORDER_ROBUSTNESS_TEMPLATE_KEY] #todo
            prompt = prompt_template["prompt"]
            options_format = prompt_template["options_format"]
    except FileNotFoundError:
        eval_logger.error("Prompt templates not found")
        sys.exit()

    def repeat_doc_swap_correct_answer(batched_docs):

        n = len(LABELS)
        initial_len = len(next(iter(batched_docs.values())))
        keys = list(batched_docs.keys())
        new_batched_docs = {key: [] for key in keys}
        new_batched_docs["always_same_option"] = []
        new_batched_docs["prompt"] = []
        new_batched_docs["options_format"] = []
        new_batched_docs["original_answer_index"] = []
        
        for doc_ind in range(initial_len):
            for label_ind, label in enumerate(LABELS):
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
    upper_case = string.ascii_uppercase
    lower_case = string.ascii_lowercase
    numerals = ["1", "2", "3", "4", "5"]
    roman_numerals = ["I", "II", "III", "IV", "V"]
    prompt = doc["prompt"]
    options_format = doc["options_format"]
    options = "".join([options_format.format(letter=upper_case[i], 
                                             option=doc['options'][i], 
                                             numeral=numerals[i],
                                             roman_numeral=roman_numerals[i],
                                             lower_case_letter=lower_case[i]) for i in range(len(doc["options"]))])
    return prompt.format(question=doc[QUESTION_KEY], options=options)

def __postprocess_pred(pred):
    if "the best answer is" not in pred.lower():
        return pred
    pred_proc = pred.lower().split("the best answer is ")[-1].split(' ')[0]
    pred_proc = re.sub(r"[^a-zA-Z0-9]", "", pred_proc).strip()
    return pred_proc.upper()

def prompt_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0])
    final_answer = translate_model_answer_to_labels(final_answer, option_format=doc["options_format"])
    gt = doc["answer"]
    prompt_id = doc["prompt_id"]
    question_id = doc["question_id"]
    return {
                f"prompt_{prompt_id}_macro_accuracy": (question_id, prompt_id, final_answer, gt),
                "prompt_consistency_rate": (question_id, prompt_id, final_answer, gt)
            }


def option_order_robustness_process_results(doc, results) -> Dict[str, float]:
    final_answer = __postprocess_pred(results[0])
    final_answer = translate_model_answer_to_labels(final_answer, option_format=doc["options_format"])
    gt = doc["answer"]
    always_same_option = doc["always_same_option"]
    question_id = doc["question_id"]
    original_answer_index = doc["original_answer_index"]
    answer_index = doc["answer_index"],
    return {
                f"per_option_macro_accuracy_{always_same_option}": (question_id, always_same_option, final_answer, gt),
                "options_consistency_rate": (question_id, always_same_option, final_answer, original_answer_index, answer_index)
            }


def translate_model_answer_to_labels(answer, option_format=None):
    numerals = ["1", "2", "3", "4", "5"]
    roman_numerals = ["I", "II", "III", "IV", "V"]
    labels = ['A', 'B', 'C', 'D', 'E']
    answer = answer.upper()

    if option_format is None:
        return answer
    
    elif "numeral" in option_format:
        
        if "roman" in option_format:
            if answer not in roman_numerals:
                return answer
            else:
                return labels[roman_numerals.index(answer)]
            
        if answer not in numerals:
            return answer
        else:
            return labels[numerals.index(answer)]
        
    return answer



def per_prompt_macro_accuracy(results: List[Dict[str, Any]], p_id=0) -> float:
    accuracies = []
    for result in results:
        question_id, prompt_id, final_answer, gt = result
        if prompt_id != p_id:
            continue
        accuracies.append(final_answer == gt)
    
    accuracie = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {prompt_id} accuracy: {accuracie}")
    
    return np.round(accuracie, 4)


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


def per_option_macro_accuracy(results: List[Dict[str, Any]], always_opt='a') -> float:
    accuracies = []
    for result in results:
        question_id, always_same_option, final_answer, gt = result
        if always_opt != always_same_option:
            continue
        accuracies.append(int(final_answer == gt))
    
    accuracie = sum(accuracies) / len(accuracies)
    eval_logger.info(f"Prompt - {always_opt.upper()} accuracy: {accuracie}")
    
    return np.round(accuracie, 4)


per_option_macro_accuracy_a = partial(per_option_macro_accuracy, always_opt='A')
per_option_macro_accuracy_b = partial(per_option_macro_accuracy, always_opt='B')
per_option_macro_accuracy_c = partial(per_option_macro_accuracy, always_opt='C')
per_option_macro_accuracy_d = partial(per_option_macro_accuracy, always_opt='D')
per_option_macro_accuracy_e = partial(per_option_macro_accuracy, always_opt='E')
per_option_macro_accuracy_f = partial(per_option_macro_accuracy, always_opt='F')
per_option_macro_accuracy_g = partial(per_option_macro_accuracy, always_opt='G')
per_option_macro_accuracy_h = partial(per_option_macro_accuracy, always_opt='H')
per_option_macro_accuracy_i = partial(per_option_macro_accuracy, always_opt='I')
per_option_macro_accuracy_j = partial(per_option_macro_accuracy, always_opt='J')


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
