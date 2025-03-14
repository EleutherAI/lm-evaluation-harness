import datasets


def doc_to_text(doc) -> str:
    option_choices = doc["options"]
    answers = "".join((f"{k}. {v}\n") for k, v in option_choices.items())
    return f"Question: {doc['question']}\n{answers}Answer:"


def doc_to_target(doc) -> str:
    # answer_idx is `A`, `B`, `C`, `D` etc.
    return doc["answer_idx"]


def filter_dataset(dataset: datasets.Dataset, lang: str) -> datasets.Dataset:
    return dataset.filter(lambda example: example["language"].startswith(lang))


def filter_french(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "fr")


def filter_english(dataset: datasets.Dataset) -> datasets.Dataset:
    return filter_dataset(dataset, "en")
