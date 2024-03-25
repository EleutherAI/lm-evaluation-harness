def doc_to_text(doc) -> str:
    # Construct the options string
    option_choices = doc["options"]
    answers = "".join(f"{k}. {v}\n" for k, v in option_choices.items())
    # Create the prompt with the report and the options
    return f"{doc['report']}\nQuestion: Does the report mention a personal history of cancer?\n{answers}Answer:"

def doc_to_target(doc) -> str:
    # Return the correct answer's index/label (e.g., 'A')
    return doc["answer_idx"]
