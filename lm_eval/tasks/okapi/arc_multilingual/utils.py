import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # breakpoint()
        out_doc = {
            "id": doc["id"],
            "query": "Question: " + preprocess(doc["instruction"]) + "\nAnswer:",
            "choices": [
                preprocess(doc["option_a"]),
                preprocess(doc["option_b"]),
                preprocess(doc["option_c"]),
                preprocess(doc["option_d"]),
                preprocess(doc["option_e"]),
            ],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_process_doc)
