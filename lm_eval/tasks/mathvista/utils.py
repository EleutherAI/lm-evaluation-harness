import os
import re
from typing import Optional

import requests

# from api_model import make_concurrent_requests
from Levenshtein import distance


API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY is not None, (
    "Please set the OPENAI_API_KEY environment variable to use the MathVista task."
)
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4"

# required for external LM call

DEMO_PROMPT = """
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


# Function to send a single request to the OpenAI API
def send_request(prompt: str):
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 100,
        }
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        print(f"An error occurred while requesting: {e}")
        return None


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def verify_extraction(extraction):
    extraction = extraction.strip()
    if not extraction:
        return False
    return True


# taken from https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py
def get_most_similar(prediction: str, choices: list) -> float:
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


# taken from https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py
def normalize_extracted_answer(
    extraction: str,
    choices: list,
    question_type: str,
    answer_type: str,
    precision,
    ignore_empty_extractions=True,
) -> Optional[str]:
    """
    Normalize the extracted answer to match the answer type
    """

    if question_type == "multi_choice":
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
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord("A") + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == "integer":
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "float":
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "list":
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception:
        return False


def extract_answer(response: str, problem: dict, quick_extract=True) -> str:
    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem["choices"]
    query = problem["query"]
    pid = problem["pid"]

    if response == "":
        return ""

    if question_type == "multi_choice" and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception:
            pass

    # quick extraction
    if quick_extract:
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(DEMO_PROMPT, query, response)
        extraction = send_request(full_prompt)
        return extraction
    except Exception:
        print(
            f"Error in extracting answer for problem: {pid} with response: {response}"
        )

    return ""


def extract_all_answers(
    resps: list[list[str]], docs: dict, quick_extract=True
) -> list[str]:
    return [
        extract_answer(resp[0], doc, quick_extract=quick_extract)
        for resp, doc in zip(resps, docs)
    ]


# adapted from https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py
def process_results(doc: dict, results: list[str]):
    response = results[0]  # noqa: F841
    choices = doc["choices"]
    question_type = doc["question_type"]
    answer_type = doc["answer_type"]
    precision = doc["precision"]  # noqa: F841
    answer = doc["answer"]
    # step 1: extract the answer from the model response
    # extracted_answer = extract_answer(response, doc)
    extracted_answer = response
    if verify_extraction(extracted_answer):
        normalized_extraction = normalize_extracted_answer(
            extracted_answer, choices, question_type, answer_type, precision
        )
        res = safe_equal(normalized_extraction, answer)
    else:
        res = False
    return {"acc": 1.0} if res else {"acc": 0.0}


### MathVista MCQ ###


def process_docs_mcq(dataset):
    return dataset.filter(lambda x: x["question_type"] == "multi_choice")
