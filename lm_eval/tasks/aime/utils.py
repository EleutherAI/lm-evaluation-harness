import re
from typing import Dict, List, Tuple
from math_verify import parse, verify
from collections import Counter


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    responses = results[0]
    retvals = []
    answers = []
    for response in responses:

        answer = parse(response)
        # Check if answer matches target
        answer_key = next(k for k in doc.keys() if k.lower() == "answer")
        target = str(doc[answer_key])
        retval = 0
        if verify(
                parse(f'${str(target)}$'),
                answer
        ):
            retval = 1
        try:
            answers.append(answer[-1])
        except:
            answers.append(None)
        retvals.append(retval)
    
    mode, model_index = majority_voting(answers)
    mode_val = verify(parse(f'${str(mode)}$'), parse(f'${str(target)}$'))

    return {"pass@1": retvals, "pass@k": retvals, "maj@k": mode_val}

def majority_voting(lst: List[str]) -> Tuple[str, int]:
    """
    Return the most frequent item in the list and its index.

    Args:
        lst (List[Any]): List of items.

    Returns:
        Tuple[Any, int]: Most frequent item and its index (any valid occurrence).
    """

    frequency = Counter(lst)
    most_freq_item, _ = frequency.most_common(1)[0]
    index = lst.index(most_freq_item)

    return most_freq_item, index
