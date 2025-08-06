import ast


def doc_to_choice(doc):
    """Parse options string to list of choices."""
    return ast.literal_eval(doc["options"])
