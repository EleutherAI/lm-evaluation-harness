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


def extract_text(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: (
            "CSAT_korean_22" in example["id"]
            or (
                "CSAT_korean_23" in example["id"]
                and int(example["id"].split("_")[-1]) < 35
            )
            or ("TK" in example["id"] and int(example["id"].split("_")[-1]) > 4)
        )
    )


def extract_grammar(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: (
            (
                "CSAT_korean" in example["id"]
                and (
                    int(example["id"].split("_")[2]) < 21
                    and int(example["id"].split("_")[3]) > 10
                )
            )
            or (
                "Kedu_1" in example["id"]
                and (
                    example["id"].split("_")[1] != "16"
                    or not (
                        "대화" in example["question"]
                        or "발화" in example["question"]
                        or "질의" in example["question"]
                    )
                )
            )
            or ("TK" in example["id"] and int(example["id"].split("_")[-1]) < 5)
        )
    )


def extract_function(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: (
            (
                "CSAT_korean" in example["id"]
                and (
                    int(example["id"].split("_")[-1]) > 34
                    or (
                        int(example["id"].split("_")[2]) < 21
                        and int(example["id"].split("_")[3]) < 11
                    )
                )
            )
            or (
                "Kedu_16" in example["id"]
                and (
                    "대화" in example["question"]
                    or "발화" in example["question"]
                    or "질의" in example["question"]
                )
            )
            or "PSE_korean" in example["id"]
        )
    )
