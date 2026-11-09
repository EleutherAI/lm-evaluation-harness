import re

import datasets


def preprocess(text):
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
                preprocess(option)
                for option in [
                    doc["option_a"],
                    doc["option_b"],
                    doc["option_c"],
                    doc["option_d"],
                    doc["option_e"],
                ]
                if option
            ],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_process_doc)
