import json
import re

import datasets


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_docs_zh(dataset: datasets.Dataset) -> datasets.Dataset:
    # Normalize bilingual endings before map writes JSON fields into Arrow columns.
    def _parse_doc(row):
        doc = json.loads(row["text"])
        doc["endings"] = [
            ending.get("zh") if isinstance(ending, dict) else ending
            for ending in doc["endings"]
        ]
        if not all(isinstance(ending, str) for ending in doc["endings"]):
            raise ValueError(f"Invalid Chinese endings in sample {doc['id']!r}")
        return doc

    dataset = dataset.map(_parse_doc, remove_columns=["text"])
    return process_docs(dataset)
