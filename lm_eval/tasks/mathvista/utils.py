import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["question_type"].strip() == "multi_choice")
