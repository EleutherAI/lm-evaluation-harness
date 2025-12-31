def doc_to_text(doc):
    return doc["prompt"].strip() + "\nSagot:"


def doc_to_choice(doc):
    # Use relevant_answers as the option set (list of strings)
    return [a.strip() for a in doc["relevant_answers"]]

def doc_to_target(doc):
    # Return the index of the best answer in the choice list
    # (required format for multiple_choice):contentReference[oaicite:3]{index=3}
    best = doc["best_answer"].strip()
    choices = [a.strip() for a in doc["relevant_answers"]]
    return choices.index(best)
