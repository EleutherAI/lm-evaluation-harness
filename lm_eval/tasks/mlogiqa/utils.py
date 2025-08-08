import ast


def doc_to_choice(doc):
    """Parse options string to list of choices."""
    return ast.literal_eval(doc["options"])


def doc_to_text(doc):
    """Parse options and render the text template."""
    # Parse the options string into a list
    options = ast.literal_eval(doc["options"])

    # Build the text template
    text = f"Passage: {doc['context'].strip()}\n"
    text += f"Question: {doc['question'].strip()}\n"
    text += "Choices:\n"
    text += f"A. {options[0]}\n"
    text += f"B. {options[1]}\n"
    text += f"C. {options[2]}\n"
    text += f"D. {options[3]}\n"
    text += "Please choose the most suitable one among A, B, C and D as the answer to this question."

    return text
