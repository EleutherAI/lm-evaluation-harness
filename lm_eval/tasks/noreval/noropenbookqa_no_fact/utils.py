import datasets


def filter_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda example: len(example["fact"]) > 0)
