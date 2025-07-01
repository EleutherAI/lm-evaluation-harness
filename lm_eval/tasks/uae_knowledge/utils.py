def doc_to_target(doc):
    return doc["correct_answer"][0]

def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

def filter(doc):
    """
    Filter out documents that do not have a correct answer.
    """
    return bool(doc["correct_answer"]) and bool(doc["options"])