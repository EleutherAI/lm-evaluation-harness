def doc_to_text(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def gold_alias(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]
