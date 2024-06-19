import evaluate as hf_evaluate


pass_at_k = hf_evaluate.load("code_eval")

# run simple test to check code execution is enabled before model generation
test_cases = ["assert add(2, 3)==5"]
candidates = [["def add(a,b): return a*b"]]
results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])


def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


def build_references(doc):
    return doc["test"] + "\n" + f"check({doc['entry_point']})"


def build_predictions(resps, docs):
    preds = []
    for resp, doc in zip(resps, docs):
        pred = [doc["prompt"] + r for r in resp]
        preds.append(pred)

    return preds
