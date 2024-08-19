import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # modifies the contents of a single
        # document in our dataset.
        answer_list = ["A", "B", "C", "D"]
        choices = [doc["A"], doc["B"], doc["C"], doc["D"]]
        if doc.get("E", None):
            answer_list.append("E")
            choices.append(doc["E"])
        if doc.get("F", None):
            answer_list.append("F")
            choices.append(doc["F"])
        out_doc = {
            "questions": doc["question"],
            "choices": choices,
            "goal": answer_list.index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_helper)  # returns back a datasets.Dataset object
