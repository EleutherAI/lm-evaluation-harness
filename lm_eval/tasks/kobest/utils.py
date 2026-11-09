from datasets import Dataset


def copa_doc_to_text(doc: dict) -> str:
    connector = {"원인": " 왜냐하면", "결과": " 그래서"}[doc["question"].strip()]
    return f"""{doc["premise"]} {connector}"""


def copa_doc_to_target(doc: dict) -> str:
    correct_choice = doc["alternative_1"] if doc["label"] == 0 else doc["alternative_2"]
    return f"""{correct_choice}"""


def copa_doc_to_choice(doc: dict) -> list:
    return [f"""{doc["alternative_1"]}""", f"""{doc["alternative_2"]}"""]


def sentineg_doc_to_text(doc: dict):
    return f"""문장: {doc["sentence"]} 긍부정:"""


def wic_doc_to_text(doc: dict) -> str:
    return f"""문장1: {doc["context_1"]} 문장2: {doc["context_2"]} 두 문장에서 {doc["word"]}가 같은 뜻으로 쓰였나?"""


def hellaswag_process_doc(doc: Dataset) -> Dataset:
    def preprocessor(dataset):
        return {
            "query": f"""문장: {dataset["context"]}""",
            "choices": [
                dataset["ending_1"],
                dataset["ending_2"],
                dataset["ending_3"],
                dataset["ending_4"],
            ],
            "gold": int(dataset["label"]),
        }

    return doc.map(preprocessor)


def macro_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore
