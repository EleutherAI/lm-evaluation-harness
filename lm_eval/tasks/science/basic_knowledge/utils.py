def process_college_chemistry(dataset):
    return dataset.filter(lambda x: x["subject"] == "college_chemistry")

