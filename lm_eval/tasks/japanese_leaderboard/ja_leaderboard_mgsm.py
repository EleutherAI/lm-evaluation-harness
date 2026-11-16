import re


_INVALID_ANSWER = "[invalid]"

_ANSWER_REGEX = re.compile(r"(\-?[0-9\.\,]+)")


def _extract_answer(completion):
    matches = _ANSWER_REGEX.findall(completion)
    if matches:
        match_str = matches[-1].strip(".")
        match_str = match_str.replace(",", "")
        try:
            match_float = float(match_str)
        except ValueError:
            return _INVALID_ANSWER

        if match_float.is_integer():
            return int(match_float)

    return _INVALID_ANSWER


def process_results(doc, results):
    assert len(results) == 1, (
        f"results should be a list with 1 str element, but is {results}"
    )

    completion = results[0]
    extracted_answer = _extract_answer(completion)
    answer = doc["answer_number"]
    acc = extracted_answer == answer
    return {
        "acc": acc,
    }
