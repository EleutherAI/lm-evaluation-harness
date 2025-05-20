import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    label = ["A", "B", "C", "D"]

    def _process_doc(doc):
        choices = doc["choices"]
        choices["label"] = label
        answerKey = doc["answerKey"]
        if answerKey not in label:
            answerKey = label[int(answerKey) - 1]
        return {
            "question": doc["question"],
            "choices": choices,
            "answerKey": answerKey,
        }

    return dataset.filter(lambda x: len(x["choices"]["label"]) == 4).map(_process_doc)
