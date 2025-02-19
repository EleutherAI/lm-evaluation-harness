from datasets import Dataset

def extract_economy(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "economy" in example["id"].lower()
    )

def extract_geography(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "geography" in example["id"].lower()
    )

def extract_history(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "KHB" in example["id"] or "histroy" in example["id"].lower()
    )

def extract_law(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "law" in example["id"].lower() or "PSAT" in example["id"]
    )

def extract_politics(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "politics" in example["id"].lower()
    )

def extract_kpop(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "popular" in example["id"].lower()
    )

def extract_society(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "society" in example["id"].lower()
    )

def extract_tradition(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "tradition" in example["id"].lower()
    )