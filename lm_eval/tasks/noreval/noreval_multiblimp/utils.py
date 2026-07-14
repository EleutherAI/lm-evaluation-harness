def filter_dataset_1_2(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 2")

def filter_dataset_2_1(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "2 -> 1")

def filter_dataset_3_1(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "3 -> 1")

def filter_dataset_1_3(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 3")

def filter_dataset_3_2(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "3 -> 2")

def filter_dataset_2_3(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "2 -> 3")

def filter_dataset_1_23(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "1 -> 2|3")

def filter_dataset_sg_du(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "SG -> DU")

def filter_dataset_du_sg(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "DU -> SG")

def filter_dataset_sg_pl(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "SG -> PL")

def filter_dataset_pl_sg(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "PL -> SG")

def filter_dataset_pl_du(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "PL -> DU")

def filter_dataset_du_pl(dataset):
    return dataset.filter(lambda example: example["feature_vals"] == "DU -> PL")

