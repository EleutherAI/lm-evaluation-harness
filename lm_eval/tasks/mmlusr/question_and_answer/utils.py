import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # The published CSVs carry no header row and store the answer as a
        # letter, so the columns are named in the task config and used as-is.
        # A handful of rows upstream have an empty choice, which the CSV
        # reader hands back as None.
        out_doc = {
            "choices": [
                doc["choice1"] or "",
                doc["choice2"] or "",
                doc["choice3"] or "",
                doc["choice4"] or "",
            ],
            "answer": (doc["answer"] or "").strip(),
        }
        return out_doc

    return dataset.map(_helper)
