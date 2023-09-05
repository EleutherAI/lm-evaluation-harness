import datasets

def doc_to_text(doc):
    choices = "\n".join([f"({k}) {v}" for i, (k, v) in enumerate(doc["options"].items())])
    return f"Question: {doc['question']}\n{choices}\nAnswer: ("

def doc_to_target(doc):
    return "{}".format(doc["answer_idx"])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    #dataset = dataset.select(range(1))
    return dataset

def gold_alias(doc):
    return ord(doc["final_decision"]) - ord("A")