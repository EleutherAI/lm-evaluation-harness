PROMPT = "ده سؤال اختيار متعدد (مع إجابته) عن {}\n\n{}\n{}\nالاجابة:"


alpha = ["1.", "2.", "3.", "4.", "5."]


def doc_to_text(doc):

    subject = doc["subject_egyptian"]
    question = (
        doc["question"]
        if doc["context"] == ""
        else f"{doc['context']}\n\n{doc['question']}"
    )

    options = []
    
    for i, opt in enumerate(eval(str(doc["choices"]))):
        options.append(f"{alpha[i]} {opt}")
    doc_text = PROMPT.format(subject, question, "\n".join(options))
    return doc_text

def doc_to_choice(doc):
    return [alpha[i][0] for i in range(len(eval(str(doc["choices"]))))]
