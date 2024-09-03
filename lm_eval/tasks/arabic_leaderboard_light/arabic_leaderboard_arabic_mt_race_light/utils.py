import datasets
import numpy as np


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        question = doc["query"]
        answer_index = int(doc["label"])
        # Dynamically determining the choices by excluding '__few_shots', 'query' and 'label'
        choices_keys = [
            key for key in doc.keys() if key not in ["query", "label", "__few_shots"]
        ]
        choices = [doc[key] for key in choices_keys]

        instruction = "الأسئلة التالية هي أسئلة متعددة الإختيارات مع الجواب الصحيح\n\n"
        query = f"{instruction}السؤال: {question}\n"
        for index, choice in enumerate(choices):
            query += f"{index}) {choice}\n"
        query += "الإجابة:"

        return {"query": query, "choices": choices, "gold": answer_index}

    return dataset.map(_process_doc)
