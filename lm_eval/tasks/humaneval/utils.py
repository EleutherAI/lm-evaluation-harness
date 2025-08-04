import evaluate as hf_evaluate
import re

try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + clean_text(r.replace("```python\n", "")) for r in resp] for resp, doc in zip(resps, docs)]

def clean_text(text: str) -> str:
    return re.sub(r'\n(▁+)', lambda m: '\n' + ' ' * len(m.group(1)), text)

def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (clean_text(r) if r.find("```") == -1 else clean_text(r[: r.find("```")]))
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
