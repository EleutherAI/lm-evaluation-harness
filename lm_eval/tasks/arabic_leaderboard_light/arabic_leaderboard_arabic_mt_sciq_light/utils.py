import random

import datasets
import numpy as np


def doc_to_text(doc):
    instruction = (
        "بناءً على السياق أدناه، اختر الإجابة الصحيحة للسؤال التالي من قائمة الاقتراحات"
    )
    support = doc["support"]
    question = doc["question"]
    query = f"""{instruction}
    السياق:
    {support}
    السؤال:
    {question}
    الإجابات المحتملة:

    """
    return query


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        correct_answer = doc["correct_answer"]
        choices = [
            doc["distractor1"],
            doc["distractor2"],
            doc["distractor3"],
            correct_answer,
        ]

        # Shuffle the choices
        random.shuffle(choices)

        answer_index = choices.index(correct_answer)

        return {"query": doc_to_text(doc), "choices": choices, "gold": answer_index}

    return dataset.map(_process_doc)
