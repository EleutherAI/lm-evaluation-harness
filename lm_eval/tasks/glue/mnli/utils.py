def doc_to_text(doc) -> str:
    return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
        doc["premise"],
        doc["hypothesis"].strip()
        + ("" if doc["hypothesis"].strip().endswith(".") else "."),
    )
