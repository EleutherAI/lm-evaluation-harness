def process_docs(dataset):
    return dataset.filter(lambda example: not example["is_bad_source"])
