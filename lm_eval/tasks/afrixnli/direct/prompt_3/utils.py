def doc_to_target(doc):
    replacements = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return replacements[doc["label"]]
