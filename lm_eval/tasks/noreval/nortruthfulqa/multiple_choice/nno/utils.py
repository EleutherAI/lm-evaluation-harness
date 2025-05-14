def p0_nn(doc):
    prompt = "Spørsmål: {question}\n\nSvar:"
    return prompt.format(question=doc["question"])


def p1_nn(doc):
    prompt = "Spørsmål: {question}\n\nSvaralternativ:{choices}\n\nSvar:"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p2_nn(doc):
    prompt = "Spørsmål: {question}\n\nKva av følgande alternativ er rett svar på spørsmålet?{choices}"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p3_nn(doc):
    prompt = "Gitt følgande spørsmål, kva av dei moglege svara under er rett?\nSpørsmål: {question}\n{choices}"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p4_nn(doc):
    prompt = "{question}\nVel eit av følgande moglege svar:{choices}\n\nSvar:"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)
