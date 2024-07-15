from sklearn.metrics import f1_score


def doc_to_choice(doc):
    choices = eval(doc["choices"])
    return choices


def doc_to_text(doc):
    output = """You are a highly knowledgeable and intelligent artificial intelligence
                model answers multiple-choice questions about {subject}

                Question: {question}

                Choices:
                        A: {choice1}
                        B: {choice2}
                        C: {choice3}
                        D: {choice4}

                Answer:  """

    choices = eval(doc["choices"])
    text = output.format(
        subject=doc["subject"],
        question=doc["question"],
        choice1=choices[0],
        choice2=choices[1],
        choice3=choices[2],
        choice4=choices[3],
    )
    return text


def weighted_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore
