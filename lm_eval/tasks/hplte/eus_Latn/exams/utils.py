import datasets


def process_docs(dataset: datasets.Dataset):
    """Filter out examples with no answer."""

    def valid_example(example: dict) -> bool:
        """Check if an example is valid."""
        if example["answer"] not in [0, 1, 2, 3]:
            return False
        return example["candidates"] != ["", "", "", ""]

    return dataset.filter(valid_example)
