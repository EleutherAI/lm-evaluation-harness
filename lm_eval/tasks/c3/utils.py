import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Add answer index to each document."""
    def _process_doc(doc):
        # Find the index of the answer in the choice list
        answer_index = doc["choice"].index(doc["answer"])
        return {
            **doc,
            "answer_index": answer_index,
        }
    return dataset.map(_process_doc)
