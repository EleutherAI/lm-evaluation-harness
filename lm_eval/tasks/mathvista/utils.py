import os
import re
import openai
import time

from Levenshtein import distance


# NOTE: Functions in this file prepended with `mathvista_` are copied directly from mathvista's repo (https://github.com/lupantech/MathVista)

# Controls whether we use regex extraction as a first attempt or go directly to gpt-4 extraction
QUICK_EXTRACT = True


# Set extraction model
EXTRACTION_MODEL = "gpt-4o-mini"


# Copied directly from mathvista's `demo_prompt` constant (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/prompts/ext_ans.py#L5)
MATHVISTA_EXTRACT_ANSWER_DEMO_PROMPT = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

IMAGE_PROMPT_TEMPLATE = """<image>

{}"""

# ================================================
# Document-to-text and document-to-image functions
# ================================================

def doc_to_image(doc):
    return [doc["decoded_image"]]


def doc_to_text(doc):
    # We assume/set default behavior:
    # - 0 shot, so we skip demo examples. Reasoning:
    #      1. Industry standards: The majority of published results for LMMs (large multimodal models) (Qwen2.5-VL, QvQ, Llama 3.2, Llama 4, etc.) report 0-shot results.
    #      2. MathVista's paper (https://arxiv.org/pdf/2310.02255) also reports 0-shot results for LMMs, only using 2-shot results for CoT text-only models.
    # - 'solution' shot prompting, asking the model to provide a solution (rather than 'code' shot prompting (aka PoT), which asks the model to provide a Python code)
    # - no captions
    #      - Reasoning: MathVista's paper (https://arxiv.org/pdf/2310.02255) does not use captions for LMM results.
    # - no OCR
    #      - Reasoning: MathVista's paper (https://arxiv.org/pdf/2310.02255) does not use OCR for LMM results.
    problem = {
        "question": doc["question"],
        "choices": doc["choices"],
        "unit": doc["unit"],
        "ocr": "", # no OCR
        "caption": "", # no captions
        "precision": doc["precision"],
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
    }
    query = mathvista_create_one_query(
        problem=problem,
        examples=[], # 0 shot
        shot_num=0, # 0 shot
        shot_type="solution", # 'solution' shot prompting
        use_caption=False, # no captions
        use_ocr=False # no OCR
    )
    return IMAGE_PROMPT_TEMPLATE.format(query)

# Copied directly mathvista's `create_one_query` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/build_query.py#L152)
def mathvista_create_one_query(problem, examples, shot_num, shot_type, use_caption, use_ocr):
    ### [1] Demo prompt
    if shot_num == 0:
        demo_prompt = ""
    else:
        demos = []
        shot_num = min(shot_num, len(examples))
        for example in examples[:shot_num]:
            prompt = ""

            # question
            prompt += f"Question: {example['question']}"

            # choices
            if "choices" in example:
                texts = ["Choices:"]
                for i, choice in enumerate(example['choices']):
                    texts.append(f"({chr(ord('A')+i)}) {choice}")
                prompt += "\n" + "\n".join(texts)

            # caption
            if use_caption:
                caption = example['caption'] if 'caption' in example else ""
                if caption != "":
                    prompt += "\n" + f"Image description: {caption}"

            # ocr
            if use_ocr:
                ocr = example['ocr'] if 'ocr' in example else ""
                if ocr != "":
                    prompt += "\n" + f"Image detected text: {ocr}"

            # solution
            if shot_type == 'solution':
                solution = example['solution'].strip()
                prompt += "\n" + f"Solution: {solution}"

            # code
            if shot_type == 'code':
                code = example['code'].strip()
                prompt += "\n" + f"Python code: {code}"

            demos.append(prompt)

        demo_prompt = "\n\n".join(demos)

    ### [2] Test query
    # problem info
    question = problem['question']
    unit = problem['unit']
    choices = problem['choices']
    caption = problem['caption']
    ocr = problem['ocr']
    precision = problem['precision']
    question_type = problem['question_type']
    answer_type = problem['answer_type']

    # hint
    if shot_type == 'solution':
        if question_type == "multi_choice":
            assert answer_type == "text"
            hint_text = f"Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end."
        else:
            assert answer_type in ["integer", "float", "list"]
            if answer_type == "integer":
                hint_text = f"Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end."

            elif answer_type == "float" and precision == 1:
                hint_text = f"Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end."

            elif answer_type == "float" and precision == 2:
                hint_text = f"Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end."

            elif answer_type == "list":
                hint_text = f"Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end."
    else:
        assert shot_type == 'code'
        hint_text = "Hint: Please generate a python code to solve the problem"

    # question
    question_text = f"Question: {question}"
    if unit:
        question_text += f" (Unit: {unit})"

    # choices
    if choices:
        # choices: (A) 1.2 (B) 1.3 (C) 1.4 (D) 1.5
        texts = ["Choices:"]
        for i, choice in enumerate(choices):
            texts.append(f"({chr(ord('A')+i)}) {choice}")
        choices_text = "\n".join(texts)
    else:
        choices_text = ""

    # caption
    caption_text = ""
    if use_caption and caption != "":
        caption_text = f"Image description: {caption}"

   # ocr
    ocr_text = ""
    if use_ocr and ocr != "":
        ocr_text = f"Image detected text: {ocr}"

    # prompt
    if shot_type == 'solution':
        prompt = "Solution: "
    else:
        assert shot_type == 'code'
        prompt = "Python code: "

    elements = [question_text, choices_text, caption_text, ocr_text, hint_text, prompt]
    test_query = "\n".join([e for e in elements if e != ""])

    ### [3] Final query
    query = demo_prompt + "\n\n" + test_query
    query = query.strip()
    return query


# ================================================
# Results processing functions
# ================================================

def process_results(doc, results):
    is_correct = False
    if results and results[0]:
        response = results[0]
        extracted_answer = mathvista_extract_answer(
            doc=doc,
            response=response
        )
        normalized_extracted_answer = mathvista_normalize_extracted_answer(
            extraction=extracted_answer,
            choices=doc['choices'],
            question_type=doc['question_type'],
            answer_type=doc['answer_type'],
            precision=doc['precision']
        )
        if mathvista_safe_equal(
            prediction=normalized_extracted_answer,
            answer=doc['answer']
        ):
            is_correct = True

    return {
        "acc": float(is_correct),
    }




# Leverages mathvista's `extract_answer` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/extract_answer.py#L29)
def mathvista_extract_answer(doc, response):
    question_type = doc['question_type']
    answer_type = doc['answer_type']
    choices = doc['choices']
    query = doc['query']
    pid = doc['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    # NOTE: edited from mathvista's `extract_answer` function to use the global QUICK_EXTRACT variable rather than a function argument
    if QUICK_EXTRACT:
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception:
            pass

    try:
        full_prompt = mathvista_create_test_prompt(MATHVISTA_EXTRACT_ANSWER_DEMO_PROMPT, query, response)
        extraction = get_extraction_response(extraction_prompt=full_prompt)
        return extraction
    except Exception:
        pass

    return ""


# Copied directly from mathvista's `verify_extraction` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/extract_answer.py#L15)
def mathvista_verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


# Copied directly from mathvista's `create_test_prompt` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/extract_answer.py#L22)
def mathvista_create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


# Based off of lmms-eval mathvista.mathvista_eval.MathvistaEvaluator.get_chat_response (https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/4ca1a52b55ac4d057329bb1dde092ce68b60256e/lmms_eval/tasks/mathvista/mathvista_evals.py#L183)
def get_extraction_response(extraction_prompt, temperature=0, max_tokens=256, n=1, patience=5, sleep_time=0):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = openai.OpenAI(api_key=openai_api_key)
        messages = [
            {"role": "user", "content": extraction_prompt},
        ]
        payload = {"model": EXTRACTION_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}

        while patience > 0:
            patience -= 1
            try:
                response = client.chat.completions.create(**payload)
                if n == 1:
                    prediction = response.choices[0].message.content.strip()
                    if prediction and prediction != "":
                        return prediction
                else:
                    prediction = [choice.message.content.strip() for choice in response.choices]
                    if prediction and prediction[0] != "":
                        return prediction

            except Exception:
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return ""


# Copied directly from mathvista's `get_most_similar` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/calculate_score.py#L17)
def mathvista_get_most_similar(prediction, choices):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


# Copied directly from mathvista's `normalize_extracted_answer` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/calculate_score.py#L27)
def mathvista_normalize_extracted_answer(
    extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False
):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == 'multi_choice':
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # if the extraction is empty, return None
        if ignore_empty_extractions and not extraction:
            return None

        # extract "A" from "(A) text"
        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = mathvista_get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


# Copied directly from mathvista's `safe_equal` function (https://github.com/lupantech/MathVista/blob/ece407d305f0bca51b689567c956cf70c8cdd847/evaluation/calculate_score.py#L84)
def mathvista_safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        return False