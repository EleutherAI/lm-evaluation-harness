def doc_to_text(doc):
    inputs = " ".join(doc["code_tokens"]).replace("\n", " ")
    inputs = " ".join(inputs.strip().split())

    return inputs


def doc_to_target(doc):
    targets = " ".join(doc["docstring_tokens"]).replace("\n", "")
    targets = " ".join(targets.strip().split())

    return targets
