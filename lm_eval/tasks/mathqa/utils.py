import re


def doc_to_choice(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices


def doc_to_target(doc):
    choices = doc_to_choice(doc)
    return choices[["a", "b", "c", "d", "e"].index(doc["correct"])]
