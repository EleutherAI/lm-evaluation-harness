import ast
import os
import random
import re
from typing import List

import numpy as np


try:
    from openai import OpenAI
except ImportError:
    pass

API_TYPE = os.getenv("API_TYPE", "openai")
MODEL_VERSION = os.getenv("MODEL_VERSION", "gpt-4.1-mini")

SYSTEM_PROMPT = """You are an impartial grader for multiple-choice questions.
You are given:
1) the model's free-form output (student_answer),
2) the available options (each with a letter and text),
3) the correct answer (by letter and/or text).

Your job:
- Extract which single option the student intended (by letter if present, otherwise by best semantic match to the option text).
- Compare that choice to the correct answer.
- Output only a single character: 1 if correct, 0 if incorrect.
No explanation. No extra characters. Just 1 or 0."""


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "<image>"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(
        [
            f"{option_letter}. {option}"
            for option_letter, option in zip(option_letters, options)
        ]
    )
    return choices_str


def construct_prompt(doc, mc_prompt=""):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(f"{doc['choices']}"))
    question = f"Given the provided image <image>, answer following questions:\n{question}\n{parsed_options}\n\n{mc_prompt}"
    return question


def pisa_doc_to_text(doc):
    question = construct_prompt(doc)
    return question


def pisa_doc_to_visual(doc):
    image_key = "image"
    if doc[image_key] is None:
        return None
    return [doc[image_key]]


def pisa_process_results(doc, results, **kwargs):
    """Default evaluation of answers based on substring matching."""
    index2ans, all_choices = get_multi_choice_info(
        ast.literal_eval(f"{doc['choices']}")
    )
    parsed_pred = parse_multi_choice_response(results[0], all_choices, index2ans)
    gold_i = doc["answer"]
    pred_i = all_choices.index(parsed_pred) if parsed_pred in all_choices else None
    is_correct = gold_i == pred_i if pred_i is not None else False

    return {
        "acc": float(is_correct),
    }


def pisa_process_results_llm_judged(doc, results, **kwargs):
    """Evaluation of answers based on LLM as a judge."""
    assert os.getenv("OPENAI_API_KEY") is not None, (
        "OPENAI_API_KEY environment variable is not set."
    )
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Please install openai package to use LLM judging.")

    index2ans, all_choices = get_multi_choice_info(
        ast.literal_eval(f"{doc['choices']}")
    )
    gold_i = doc["answer"]
    correct_answer = index2ans[all_choices[gold_i]]
    is_correct = (
        judge_mcq(
            results[0],
            [f"{k}) {v}" for k, v in index2ans.items()],
            f"{chr(ord('A') + gold_i)}) {correct_answer}",
        )
        == 1
    )

    return {
        "acc": float(is_correct),
    }


def eval_multi_choice(gold_i, pred_i):
    """Evaluate a multiple choice instance."""
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/pisa-Benchmark/pisa/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/pisa-Benchmark/pisa/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def extract_numbers(string):
    """
    Exact all forms of numbers from a string with regex.
    """
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    # Extract numbers with commas
    numbers_with_commas = re.findall(pattern_commas, string)
    # Extract numbers in scientific notation
    numbers_scientific = re.findall(pattern_scientific, string)
    # Extract simple numbers without commas
    numbers_simple = re.findall(pattern_simple, string)

    # Combine all extracted numbersz
    all_numbers = numbers_with_commas + numbers_scientific + numbers_simple
    return all_numbers


def check_is_number(string):
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string):
    """Normalize the str to lower case and make them float numbers if possible."""
    # check if characters in the string

    # if number, numerize it.
    string = string.strip()

    is_number = check_is_number(string)

    if is_number:
        string = string.replace(",", "")
        string = float(string)
        # leave 2 decimal
        string = round(string, 2)
        return [string]
    else:  # it's likely to be a string
        # lower it
        string = string.lower()
        if len(string) == 1:
            return [" " + string, string + " "]  # avoid trivial matches
        return [string]


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices


# LLM as a judge utils
def build_user_prompt(student_answer: str, options: List[str], correct: str) -> str:
    """
    options: like ["A) red", "B) blue", "C) green", "D) yellow"]
    correct: either a letter like "B" or the full option text. Both are provided to help you.
    """
    return f"""Student Answer:
{student_answer.strip()}

Options:
{chr(10).join(options)}

Correct Answer (letter and/or text):
{correct.strip()}

Instructions:
- If student gives multiple letters, pick the *final* one.
- If no clear letter, pick the best-matching option by meaning.
- Output only 1 or 0.
"""


def judge_mcq(pred: str, options: List[str], correct: str) -> int:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    user_prompt = build_user_prompt(pred, options, correct)

    resp = client.chat.completions.create(
        model=MODEL_VERSION,
        temperature=0,
        max_tokens=1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    return 1 if raw == "1" else 0
