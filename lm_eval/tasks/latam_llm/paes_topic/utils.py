import datasets
import ast

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "question": doc["question"],
            "subject": doc["subject"],
            "choices": ast.literal_eval(doc["choices"]),
            "answer": doc["answer"],
            "theme":doc["theme"]
        }
        return out_doc

    return dataset.map(_process_doc)
