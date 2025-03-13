import datasets


def filter_dataset(dataset: datasets.Dataset, gender: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["gender"] == gender)


def filter_male(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "male")


def filter_female(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "female")


def filter_neutral(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "neutral")
