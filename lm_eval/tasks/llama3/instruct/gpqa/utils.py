import re


def process_docs(dataset):
    """Process GPQA dataset - re-use from the parent gpqa task."""

    def _process_doc(doc):
        choices = [doc["choice1"], doc["choice2"], doc["choice3"], doc["choice4"]]
        # The correct answer index
        correct_idx = doc["answer"]
        doc["answer"] = ["A", "B", "C", "D"][correct_idx]
        return doc

    return dataset.map(_process_doc)