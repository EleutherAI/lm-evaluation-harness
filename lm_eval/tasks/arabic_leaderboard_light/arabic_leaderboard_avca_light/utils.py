import datasets


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        question = doc["question"]
        answer = doc["answer"]

        return {
            "query": f"السؤال: {question}\nالإجابة:",
            "choices": ["صح", "خطأ"],
            "gold": ["صح", "خطأ"].index(answer),
        }

    return dataset.map(_process_doc)
