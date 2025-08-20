
def get_ar_to_en_samples(dataset):
    """
    Filters the dataset to include only Arabic-English translation pairs.
    """
    return dataset.filter(lambda x: x["source"] == "ar-to-en")


def get_en_to_ar_samples(dataset):
    """
    Filters the dataset to include only English-Arabic translation pairs.
    """
    return dataset.filter(lambda x: x["source"] == "en-to-ar")
