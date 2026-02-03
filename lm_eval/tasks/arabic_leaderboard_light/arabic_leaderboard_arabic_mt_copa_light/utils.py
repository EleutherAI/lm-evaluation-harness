import datasets


def process_docs(dataset: datasets.Dataset):
    def _process_doc(doc):
        premise = doc["premise"]
        choices = [doc["choice1"], doc["choice2"]]
        question_map = {"cause": "لأن", "effect": "لذلك"}
        question = question_map[doc["question"]]
        answer = doc["label"]

        query = f"{premise}، {question} :\n0) {choices[0]}\n1) {choices[1]}\nالإجابة:"

        return {"query": query, "choices": choices, "gold": answer}

    return dataset.map(_process_doc)
