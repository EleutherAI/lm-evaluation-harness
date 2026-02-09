def doc_to_target(doc):
    replacements = {0: "True", 1: "Neither", 2: "False"}
    return replacements[doc["label"]]
