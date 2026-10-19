from datasets import Dataset


LETTERS = ["A", "B", "C", "D", "E"]


def get_context(doc) -> str:
    ctx = doc["paragraph"]
    q = doc["question"]
    opt = doc["choices"]
    letters = LETTERS[: len(opt)]
    # Keep the original layout (no space after "A:") so 4-option prompts are unchanged.
    options = ", ".join(
        f"{letter}:{o}" if i == 0 else f"{letter}: {o}"
        for i, (letter, o) in enumerate(zip(letters, opt, strict=True))
    )
    letter_list = ", ".join(letters)
    if ctx:
        res = f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 {letter_list} 중에 골라 알파벳 하나로 답하시오.\n\n맥락: {ctx}\n질문: {q}\n보기:\n{options}\n정답:"
    else:
        res = f"주어진 질문을 천천히 읽고, 적절한 정답을 {letter_list} 중에 골라 알파벳 하나로 답하시오.\n\n질문: {q}\n보기:\n{options}\n정답:"

    return res


def get_target(doc) -> str:
    return LETTERS[doc["choices"].index(doc["answer"])]


def get_choices(doc) -> list[str]:
    return LETTERS[: len(doc["choices"])]


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
