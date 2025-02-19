from datasets import Dataset

def extract_text(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example:
            "CSAT_korean_22" in example["id"] or
            ("CSAT_korean_23" in example["id"] and int(example["id"].split('_')[-1]) < 35) or
            ("TK" in example["id"] and int(example["id"].split('_')[-1]) > 4)
    )

def extract_grammar(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example:
            ("CSAT_korean" in example["id"] and 
                (int(example["id"].split('_')[2]) < 21 and int(example["id"].split('_')[3]) > 10)
            ) or
            ("Kedu_1" in example["id"] and
                (example["id"].split('_')[1] != "16" or
                 not ("대화" in example["question"] or "발화" in example["question"] or "질의" in example["question"])
                )
            ) or
            ("TK" in example["id"] and int(example["id"].split('_')[-1]) < 5)
    )

def extract_function(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example:
            ("CSAT_korean" in example["id"] and
                (int(example["id"].split('_')[-1]) < 11 or int(example["id"].split('_')[-1]) > 34)
            ) or
            ("Kedu_16" in example["id"] and
                ("대화" in example["question"] or "발화" in example["question"] or "질의" in example["question"])
            ) or
            "PSE_korean" in example["id"]
    )