import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _helper(doc):
        # The dataset's own loading script normalised the raw context before
        # yielding it, and the prompts depend on that exact shape, so keep the
        # same four steps here now that the script is read as plain jsonl.
        return {
            "context": doc["context"]
            .strip()
            .replace("\n\n", "\n")
            .replace("Q:", "Question:")
            .replace("A:", "Answer:"),
            "completion": doc["completion"],
        }

    return dataset.map(_helper)
