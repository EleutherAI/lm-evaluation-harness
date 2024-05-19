voc = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def doc_to_text(x):
    question = x["question"].strip()
    choices = x["options"]
    inp = f"{question}"
    for idx, choice in enumerate(choices):
        inp = inp + f"\\n{voc[idx]}. {choice}"
    return inp + "\\nRisposta:"


def doc_to_choice(x):
    choices = x["options"]
    answers = []
    for idx, _ in enumerate(choices):
        answers.append(voc[idx])
    return answers
