def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs, doc["QUESTION"], doc["final_decision"]
    )
