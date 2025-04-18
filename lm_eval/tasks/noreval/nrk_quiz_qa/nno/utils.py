def p0_nn(doc):
    prompt = "Spørsmål: {question}\n\nSvar:"
    return prompt.format(question=doc["question"])


def p1_nn(doc):
    prompt = "{question}\n\nSvaralternativer:{choices}\n\nKva er rett svar?\n\nSvar:"
    choices = "".join(list(map(lambda choice: f"\n- {choice}", doc["choices"]["text"])))
    return prompt.format(question=doc["question"], choices=choices)


def p2_nn(doc):
    prompt = "{question}{choices}\n\nEr det rette svaret {enumerated_choices}?\n\nSvar:"
    choices = "".join(
        [
            f"\n{label}: {option}"
            for label, option in zip(doc["choices"]["label"], doc["choices"]["text"])
        ]
    )
    enumerated_choices = ", ".join(
        doc["choices"]["label"][:-1]
    ) + ", eller {latest_choice}".format(latest_choice=doc["choices"]["label"][-1])
    if len(doc["choices"]["label"]) == 2:
        enumerated_choices = enumerated_choices.replace(", eller", " eller")
    return prompt.format(
        question=doc["question"], choices=choices, enumerated_choices=enumerated_choices
    )


def p3_nn(doc):
    prompt = "Spørsmål: {question}{choices}\n\nSvar:"
    choices = "".join(
        [
            f"\n{label}: {option}"
            for label, option in zip(doc["choices"]["label"], doc["choices"]["text"])
        ]
    )
    return prompt.format(question=doc["question"], choices=choices)


def p4_nn(doc):
    prompt = "{question}\nVel rett svar blant desse alternativa:{choices}\n\nSvar:"
    choices = "".join(list(map(lambda choice: f"\n- {choice}", doc["choices"]["text"])))
    return prompt.format(question=doc["question"], choices=choices)
