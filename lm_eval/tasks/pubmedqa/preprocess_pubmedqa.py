def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["context"]["contexts"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(
        ctxs,
        doc["question"],
    )
