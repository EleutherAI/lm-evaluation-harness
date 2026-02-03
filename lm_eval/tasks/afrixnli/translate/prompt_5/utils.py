def doc_to_target(doc):
    replacements = {0: "true", 1: "false", 2: "inconclusive"}
    return replacements[doc["label"]]
