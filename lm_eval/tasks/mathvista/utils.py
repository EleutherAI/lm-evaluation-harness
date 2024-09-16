import re

from Levenshtein import distance


# taken from https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py
def get_most_similar(prediction: str, choices: list):
    """
    Use the Levenshtein distance (or edit distance) to determine which of the choices is most similar to the given prediction
    """
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]
    # return min(choices, key=lambda choice: distance(prediction, choice))


def normalize_extracted_answer(
    extraction,
    choices: list,
    question_type: str,
    answer_type: str,
    precision,
    ignore_empty_extractions=False,
):
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


def get_acc_with_contion(res_pd, key, value):
    if key == "skills":
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd["true_false"] == True]  # noqa: E712
    acc = len(correct_pd) / len(total_pd)

    return len(correct_pd), len(total_pd), acc


# adapted from https://github.com/lupantech/MathVista/blob/main/evaluation/extract_answer.py
def process_results(doc, results):
    response = results[0]
    choices = doc["choices"]
    question_type = doc["question_type"]
    answer_type = doc["answer_type"]
    precision = doc["precision"]  # noqa: F841
    extraction = doc["extraction"]  # noqa: F841
    if question_type == "multi_choice" and response in choices:
        return {"acc": 1.0}
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
