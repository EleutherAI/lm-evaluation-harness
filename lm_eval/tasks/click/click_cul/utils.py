from typing import List

from datasets import Dataset


def get_context(doc) -> str:
    ctx = doc["paragraph"]
    q = doc["question"]
    opt = doc["choices"]
    if ctx:
        res = f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n맥락: {ctx}\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n정답:"
    else:
        res = f"주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n정답:"

    return res


def get_target(doc) -> str:
    ans = doc["answer"]
    if "CSAT" in doc["id"]:
        return ["A", "B", "C", "D", "E"][doc["choices"].index(ans)]
    return ["A", "B", "C", "D"][doc["choices"].index(ans)]


def get_choices(doc) -> List[str]:
    if "CSAT" in doc["id"]:
        return ["A", "B", "C", "D", "E"]
    return ["A", "B", "C", "D"]


def extract_economy(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "economy" in example["id"].lower())


def extract_geography(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "geography" in example["id"].lower())


def extract_history(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "KHB" in example["id"] or "history" in example["id"].lower()
    )


def extract_law(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "law" in example["id"].lower() or "PSAT" in example["id"]
    )


def extract_politics(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "politics" in example["id"].lower())


def extract_kpop(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "popular" in example["id"].lower())


def extract_society(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "society" in example["id"].lower())


def extract_tradition(dataset: Dataset) -> Dataset:
    return dataset.filter(lambda example: "tradition" in example["id"].lower())
