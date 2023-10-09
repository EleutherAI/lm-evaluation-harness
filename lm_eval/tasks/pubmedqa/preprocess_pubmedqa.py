def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs, doc["QUESTION"], doc["final_decision"]
    )


def doc_to_target(doc) -> str:
    return " {}".format(doc["final_decision"])


def gold_alias(doc):
    dict_to_label = {"yes": 0, "no": 1, "maybe": 2}
    return dict_to_label[doc["final_decision"]]
