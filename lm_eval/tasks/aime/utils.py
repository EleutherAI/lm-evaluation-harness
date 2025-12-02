import re
from typing import Dict, List, Tuple
from collections import Counter
from math_verify import parse, verify
try:
    from math_verify import parse, verify
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "`math_verify` is required for AIME tasks. "
        "Please install via `pip install lm-eval[math]` or `pip install math-verify`"
    ) from e

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
    
    correct_count = sum(retvals)
    maj_val = 1 if correct_count > len(retvals) / 2 else 0

    return {"pass@1": retvals, "pass@k": retvals, "maj@k": maj_val}
