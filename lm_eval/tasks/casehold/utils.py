import datasets
from evaluate import load

choices_keys = [
    "holding_0",
    "holding_1",
    "holding_2",
    "holding_3",
    "holding_4",
]

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        prompt = f"Complete the following excerpt from a US court opinion:\n{doc['citing_prompt']}\nAnswer:"
        choices = [doc[key] for key in choices_keys]
        out_doc = {
            "prompt": prompt,
            "choices": choices,
            "gold": int(doc["label"]),
        }
        return out_doc
    return dataset.map(_process_doc)


def micro_f1_score(items):
    f1_metric = load("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="micro")[
        "f1"
    ]
    return f1_score

def macro_f1_score(items):
    f1_metric = load("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="macro")[
        "f1"
    ]
    return f1_score