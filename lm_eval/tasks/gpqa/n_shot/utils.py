import datasets
import re


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
            # "query": "Question: " + preprocess(doc["instruction"]) + "\nAnswer:",
            "choice1": preprocess(doc["Incorrect Answer 1"]),
            "choice2": preprocess(doc["Incorrect Answer 2"]),
            "choice3": preprocess(doc["Incorrect Answer 3"]),
            "choice4": preprocess(doc["Correct Answer"]),
            "answer": "(D)"
        }
        return out_doc

    return dataset.map(_process_doc)
