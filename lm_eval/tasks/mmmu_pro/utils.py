import ast
import random
import re

import numpy as np


random.seed(42)

# ============ Direct lm_eval utility functions. Copied from mmmu/utils.py ============ 
# Link: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmmu/utils.py


# From https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/prompts.yaml

# Direct prompting 
MULTI_CHOICE_EXAMPLE_FORMAT_DIRECT = """{}

{}

Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of the options."""


MULTI_CHOICE_EXAMPLE_FORMAT_VISION_DIRECT = """<image>

Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of the options."""

# COT prompting
MULTI_CHOICE_EXAMPLE_FORMAT_COT = """{}

{}

Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of the options. Think step by step before answering."""


MULTI_CHOICE_EXAMPLE_FORMAT_VISION_COT = """<image>

Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of the options. Think step by step before answering."""


START_CHR = "A"

# input utility functions

# standard variants image input
def doc_to_image(doc):
    # get formatted prompt (incl. multi-choice options) pre-<image {i}> reformatting
    # we use _doc_to_text_direct here, but direct vs. COT doesn't matter for the images so this can be used for either variant
    input_text = _doc_to_text_direct(doc)
    # locate <image {i}> instances in input
    image_placeholders = [
        img.replace(" ", "_").replace("<", "").replace(">", "")
        for img in re.findall("<image [1-7]>", input_text)
    ]

    # collect visuals (can have dupes of a given image or be out of order)
    # E.g. validation_Math_19 contains <image 1> and <image 2> but seen as [<image 1>, <image 2>, <image 1>, <image 1>, <image 2>]
    visuals = [doc[img] for img in image_placeholders]

    return visuals


# standard variants, direct prompting
def doc_to_text_direct(doc):
    """Get the prompt for a given document."""

    prompt = _doc_to_text_direct(doc)

    for i in range(1, 8):
        # replace <image {i}> with <image>. TODO: check this is always the right decision incl. for non-HF models
        prompt = prompt.replace(f"<image {i}>", "<image>")

    return prompt


def _doc_to_text_direct(doc):
    """Helper--get the prompt for a given document but DO NOT yet replace <image {i}> with <image>."""
    choices_str = ""

    for i, choice in enumerate(ast.literal_eval(doc["options"])):
        # add (A) {choice1}\n , (B) {choice2}\n , and so on
        # to create the list of formatted choices in the prompt
        choices_str += f"\n({chr(ord(START_CHR) + i)}) {choice}"

    choices_str = (
        choices_str.lstrip()
    )  # remove the extraneous prepended \n that we added

    prompt = MULTI_CHOICE_EXAMPLE_FORMAT_DIRECT.format(doc["question"], choices_str)

    return prompt
    

# standard variants, COT prompting
def doc_to_text_cot(doc):
    """Get the prompt for a given document."""

    prompt = _doc_to_text_cot(doc)

    for i in range(1, 8):
        # replace <image {i}> with <image>. TODO: check this is always the right decision incl. for non-HF models
        prompt = prompt.replace(f"<image {i}>", "<image>")

    return prompt


def _doc_to_text_cot(doc):
    """Helper--get the prompt for a given document but DO NOT yet replace <image {i}> with <image>."""
    choices_str = ""

    for i, choice in enumerate(ast.literal_eval(doc["options"])):
        # add (A) {choice1}\n , (B) {choice2}\n , and so on
        # to create the list of formatted choices in the prompt
        choices_str += f"\n({chr(ord(START_CHR) + i)}) {choice}"

    choices_str = (
        choices_str.lstrip()
    )  # remove the extraneous prepended \n that we added

    prompt = MULTI_CHOICE_EXAMPLE_FORMAT_COT.format(doc["question"], choices_str)

    return prompt


# vision variants
def vision_doc_to_image(doc):
    # mmmu-pro-vision always has a single image under 'image'. This is independent of CoT vs. Direct prompting.
    return [doc['image']]


def vision_doc_to_text_direct(doc):
    # mmmu-pro-vision has no question, just a single image. This image contains the question and a list of choices.
    return MULTI_CHOICE_EXAMPLE_FORMAT_VISION_DIRECT


def vision_doc_to_text_cot(doc):
    # mmmu-pro-vision has no question, just a single image. This image contains the question and a list of choices.
    return MULTI_CHOICE_EXAMPLE_FORMAT_VISION_COT


# output processing utility functions
def process_results(doc, results):
    # multichoice logic
    option_strs = ast.literal_eval(doc["options"])
    index2ans, all_choices = get_multi_choice_info(option_strs)

    pred = parse_multi_choice_response(results[0], all_choices, index2ans)
    # print(pred, all_choices, index2ans)
    is_correct = eval_multi_choice(doc["answer"], pred)

    return {
        "acc": float(is_correct),
    }

# ============ MMMU-Pro utility functions, for use in the above. Copied directly from MMMU repo ============ 
# Link: https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu-pro/evaluate.py

def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
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


def parse_multi_choice_responses(response):
    pred_indexs = []
    return pred_indexs

def parse_multi_choice_response(response, all_choices, index2ans):
    """
    Parse the prediction from the generated response.
    Return the predicted index, e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()
        
        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]
        
        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

        
    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        print (response)
        response = ""
    

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


def get_multi_choice_info(options):
    """
    Given the list of options for multiple choice question
    Return the index2ans and all_choices
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/data_utils.py#L54
    """

    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))

    return index2ans, all_choices