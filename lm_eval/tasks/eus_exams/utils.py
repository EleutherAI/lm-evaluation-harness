import datasets


def process_docs(dataset: datasets.Dataset):
    """Filter out examples with no answer."""

    def valid_example(example: dict) -> bool:
        """Check if an example is valid."""
        if example["answer"] not in [0, 1, 2, 3]:
            return False
        if example["candidates"] == ["", "", "", ""]:
            return False
        return True

    return dataset.filter(valid_example)
