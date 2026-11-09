# Prompt builders referenced by the norquad_nob_p* configs.


def p0(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f"Tittel: {title}\n\nTekst: {passage}\n\nSpørsmål: {question}\n\nSvar:"
    return prompt


def p1(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nGitt teksten over, hva er svaret på følgende spørsmål? "{question}"\n\nSvar:'
    return prompt


def p2(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = (
        f"Tittel: {title}\n\nTekst: {passage}\n\nSvar på følgende: {question}\n\nSvar:"
    )
    return prompt


def p3(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nHvordan kan man svare på spørsmålet "{question}", gitt teksten over?\n\nSvar:'
    return prompt


def p4(doc):
    title = doc["title"]
    passage = doc["passage"]
    question = doc["question"]
    prompt = f'Tittel: {title}\n\nTekst: {passage}\n\nGitt teksten over, besvar følgende spørsmål: "{question}"\n\nSvar:'
    return prompt
