import datasets
import numpy as np


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        premise = doc["premise"]
        choices = [doc["choice1"], doc["choice2"]]
        question_map = {"cause": "لأن", "effect": "لذلك"}
        question = question_map[doc["question"]]
        answer = doc["label"]

        query = "{}، {} :\n0) {}\n1) {}\nالإجابة:".format(
            premise, question, choices[0], choices[1]
        )

        return {"query": query, "choices": choices, "gold": answer}

    return dataset.map(_process_doc)
