def doc_to_target(doc):
    labels = [c["label"] for c in doc["question"]["choices"]]

    try:
        i = labels.index(doc["answerKey"].lstrip())
    except Exception as e:
        print("Failed", e)
        return
    return i


def doc_to_choice(doc):
    texts = [c["text"] for c in doc["question"]["choices"]]
    return texts


def doc_to_text(doc):
    return doc["question"]["stem"].strip()
