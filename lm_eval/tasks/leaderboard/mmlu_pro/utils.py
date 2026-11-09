import string


def doc_to_text(doc):
    doc_to_text = f"{doc['question']}\n"

    for i in range(len(doc["options"])):
        doc_to_text += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"

    doc_to_text += "Answer:"
    return doc_to_text


def doc_to_choice(doc):
    return [string.ascii_uppercase[i] for i in range(len(doc["options"]))]
