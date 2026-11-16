import datasets
import numpy as np


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        question = doc["question"]
        passage = doc["passage"]
        instruction = "بناء على المقطع التالي، أجب عن السؤال ب نعم أو لا"
        query = f"""{instruction}
        المقطع :
        {passage}
        السؤال:
        {question}
        الإجابة:
        """

        return {
            "query": query,
            "choices": ["نعم", "لا"],
            "gold": 0 if doc["answer"] else 1,
        }

    return dataset.map(_process_doc)
