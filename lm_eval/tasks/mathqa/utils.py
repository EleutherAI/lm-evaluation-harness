import re


def create_choices(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices


def doc_to_target(doc):
    choices = create_choices(doc)
    return choices[["a", "b", "c", "d", "e"].index(doc["correct"])]
