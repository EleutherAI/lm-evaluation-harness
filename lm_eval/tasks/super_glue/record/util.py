def doc_to_text(doc):
    initial_text, *highlights = doc["passage"].strip().split("\n@highlight\n")
    text = initial_text + "\n\n"
    for highlight in highlights:
        text += f"  - {highlight}.\n"
    return text


def format_answer(query, entity):
    return f"  - {query}".replace("@placeholder", entity)


def doc_to_target(doc):
    # We only output the first correct entity in a doc
    return format_answer(query=doc["query"], entity=doc["answers"][0])
