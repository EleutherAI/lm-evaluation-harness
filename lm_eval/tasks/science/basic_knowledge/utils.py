def process_college_chemistry(dataset):
    return dataset.filter(lambda x: x["subject"] == "college_chemistry")

def process_hle_chemistry(dataset):
    chemistry_dataset = dataset.filter(lambda x: x["category"] == "Chemistry")
    return chemistry_dataset.filter(lambda x: x["image"] == "")