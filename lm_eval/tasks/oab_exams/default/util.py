def generate_options(choices):
    options = ""
    for text, label in zip(choices['text'], choices['label']):
        options += f"{label}) {text}\n"
    return options.strip()

def doc_to_text(doc):
    return f"Questão:\n{doc['question']}\nAlternativas:\n{generate_options(doc['choices'])}\nResposta Correta:"