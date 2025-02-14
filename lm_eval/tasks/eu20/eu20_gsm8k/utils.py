import re

def _extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def _is_correct(completion, answer):
    gold = _extract_answer(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return _extract_answer(completion) == gold

def process_results(doc, results):
    completion = results[0]
    answer = doc["answer"]
    return {"acc": _is_correct(completion, answer)}

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
