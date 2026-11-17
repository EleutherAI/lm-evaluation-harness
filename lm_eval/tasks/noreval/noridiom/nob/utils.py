def filter_dataset_nb(dataset):
    return dataset.filter(lambda example: example["language"] == "nob")
