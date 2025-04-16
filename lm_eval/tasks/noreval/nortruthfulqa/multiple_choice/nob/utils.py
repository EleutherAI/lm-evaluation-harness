def p0_nb(doc):
    prompt = "Spørsmål: {question}\n\nSvar:"
    return prompt.format(question=doc["question"])


def p1_nb(doc):
    prompt = "Spørsmål: {question}\n\nSvaralternativer:{choices}\n\nSvar:"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p2_nb(doc):
    prompt = "Spørsmål: {question}\n\nHvilke av følgende alternativer er riktig svar på spørsmålet?{choices}"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p3_nb(doc):
    prompt = "Gitt følgende spørsmål, hvilket av de mulige svarene under er riktig?\nSpørsmål: {question}\n{choices}"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)


def p4_nb(doc):
    prompt = "{question}\nVelg et av følgende mulige svar:{choices}\n\nSvar:"
    choices = "".join(
        list(map(lambda choice: f"\n- {choice}", doc["mc1_targets"]["choices"]))
    )
    return prompt.format(question=doc["question"], choices=choices)
