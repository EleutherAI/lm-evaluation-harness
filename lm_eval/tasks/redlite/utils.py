import re
import string
from rouge_score import rouge_scorer, scoring


# Input example 
# {
#   "messages": [
#     {
#       "role": "system",
#       "content": "You are a helpful problem solving assistant able to determine if a problem statement has enough information to find a solution. When you have enough information to solve a problem, you answer with \"Yes, this problem can be solved with provided information.\". If information required to solve is incomplete, unavailable or missing you answer with \"No, information is missing in order to solve this problem.\"."
#     },
#     {
#       "role": "user",
#       "content": "Here is a problem statement. Determine if it can be solved. Answer stricly with \"Yes, this problem can be solved with provided information.\" or \"No, information is missing in order to solve this problem.\" depending if you have enough information to solve or not.\nFor instance, if the problem was \"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\", your answer would be: \"Yes, this problem can be solved with provided information.\".\nConversely, if the problem was \"A paper about AI regulation that was originally submitted to arXiv.org in June 2022 shows a figure with three axes, where each axis has a label word at both ends. Which of these words is used to describe a type of society in a Physics and Society article submitted to arXiv.org on August 11, 2016?\", your answer would be: \"No, information is missing in order to solve this problem.\".\nHere's the problem statement: \"Sandra's neighbor gives her a basket of 9 eggs every time she babysits their daughter. To make a Spanish flan, she needs 3 eggs. If Sandra has been tasked to make 15 Spanish flans for her school fundraiser, how many times does Sandra have to babysit?\".\nCan it be solved?\n"
#     }
#   ],
#   "expected": "Yes, this problem can be solved with provided information.",
#   "id": "test-0"
# }

def doc_to_text(doc):
    return doc["messages"][1]["content"]

def process_results(doc, results):

    completion = results[0]
    
    pem_score = calculate_pem(doc["expected"], completion)
    rouge_score = calculate_rouge(doc["expected"], completion)
    
    return {
        "best-of-pem-rouge" : max(pem_score, rouge_score)
    }
    


def calculate_rouge(gold: str, pred: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
    scores = scorer.score(gold, pred)
    return scores["rouge2"].fmeasure

def calculate_pem(expected, actual):
    expected = normalize_string(
        expected,
        to_lower=True,
        strip_punct=True,
        normalize_whitespace=True,
    )
    actual = normalize_string(
        actual,
        to_lower=True,
        strip_punct=True,
        normalize_whitespace=True,
    )

    guarded_expected = _to_string_with_word_guards(expected)
    guarded_actual = _to_string_with_word_guards(actual)

    return 1.0 if guarded_actual.startswith(guarded_expected) else 0.0


_RE_PUNCT = "[" + re.escape(string.punctuation) + "]"

def normalize_string(
    input: str,
    *,
    to_lower=False,
    strip_articles=False,
    strip_punct=False,
    normalize_whitespace=False,
) -> str:
    """
    Normalizes string.

    - **input** (`str`): Input string to normalize
    - **to_lower** (`bool`): When set `True`, converts string to lower case. Default `False`.
    - **strip_articles** (`bool`): When set to `True` strips English articles "a", "an", "the". Default `False`.
    - **strip_punct** (`bool`): When set to `True` strips punctuation symbols (`string.punctuation`). Default `False`.
    - **normalize_whitespace** (`bool`): When set to `True` converts all whetespace to space, removes duplicate
            spaces, and strips leading and trailing space. Default `False`.

    Returns transformed string.
    """
    if to_lower:
        input = input.lower()

    if strip_articles:
        input = re.sub(r"\b(a|an|the)\b", " ", input, flags=re.IGNORECASE)

    if strip_punct:
        input = re.sub(_RE_PUNCT, "", input)

    if normalize_whitespace:
        input = re.sub(r"\s+", " ", input).strip()

    return input

def _to_string_with_word_guards(string):
    wg = "\x01"  # word guard (something that should never happen in the input string)
    assert wg not in string

    return wg + wg.join(string.split()) + wg