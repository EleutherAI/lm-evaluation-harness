def doc_to_text(doc):
    ctxs = "\n".join(doc["context"]["contexts"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs, doc["question"], doc["final_decision"]
    )


def doc_to_target(doc):
    return " {}".format(doc["final_decision"])


def gold_alias(doc):
    dict_to_label = {"yes": 0, "no": 1, "maybe": 2}
    return dict_to_label[doc["final_decision"]]
