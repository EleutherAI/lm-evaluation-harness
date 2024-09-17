import re
from typing import Optional

from Levenshtein import distance


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
            normalized_extraction = str(round(float(extraction), precision))
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


def extract_answer(response: str, problem: dict) -> str:
    question_type = problem["question_type"]
    answer_type = problem["answer_type"]
    choices = problem["choices"]
    # query = problem["query"]
    # pid = problem['pid']

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

    return ""


# adapted from https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py
def process_results(doc: dict, results: list[str]):
    response = results[0]  # noqa: F841
    choices = doc["choices"]
    question_type = doc["question_type"]
    answer_type = doc["answer_type"]
    precision = doc["precision"]  # noqa: F841
    answer = doc["answer"]
    extracted_answer = extract_answer(response, doc)
    normalized_extraction = normalize_extracted_answer(
        extracted_answer, choices, question_type, answer_type, precision
    )
    res = safe_equal(normalized_extraction, answer)
    return {"acc": 1.0} if res else {"acc": 0.0}
