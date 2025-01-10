import evaluate as hf_evaluate


# pass_at_k = hf_evaluate.load("code_eval")
#
# # run simple test to check code execution is enabled before model generation
# test_cases = ["assert add(2, 3)==5"]
# candidates = [["def add(a,b): return a*b"]]
# results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])


def pass_at_1(references: list[str], predictions: list[list[str]], k: list[int] = None):
    pass_at_k = hf_evaluate.load("code_eval")
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=k,
    )[0]

    return {
        key: val for key, val in res.items() if key in map(lambda x: f"pass@{x}", k)
    }


def build_references(doc):
    return doc["test"] + "\n" + f"check({doc['entry_point']})"


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]
