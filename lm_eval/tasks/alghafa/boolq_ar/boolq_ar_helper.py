def get_label(doc):
    if doc["answer"]:
        return 1
    else:
        return 0
