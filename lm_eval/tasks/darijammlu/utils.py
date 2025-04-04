PROMPT = "هادا سؤال متعدد الخيارات (مع الجواب ديالو) على {}\n\n{}\n{}\nالجواب:"


alpha = ["A.", "B.", "C.", "D.", "E."]


def doc_to_text(doc):
    subject = doc["subject_darija"]
    question = (
        doc["question"]
        if doc["context"] == ""
        else f"{doc['context']}\n\n{doc['question']}"
    )

    options = []
    for i, opt in enumerate(doc["choices"]):
        options.append(f"{alpha[i]} {opt}")

    doc_text = PROMPT.format(subject, question, "\n".join(options))

    return doc_text


def doc_to_choice(doc):
    return [alpha[i][0] for i in range(len(doc["choices"]))]
