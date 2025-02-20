from typing import Dict, List

from math_verify import parse, verify


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    candidates = results[0]

    # math_verify
    res = verify(parse(doc["answer"]), parse(candidates))
    mathval = 1 if res else 0

    results = {
        "math_verify": mathval,
    }
    return results
