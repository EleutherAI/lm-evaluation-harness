def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

def doc_to_target(doc):
    options = doc_to_choice(doc)
    return options.index(doc["correct_answer"][0])