import ast


def doc_to_choice(doc):
    """
    Convert a doc to a choice.
    """
    return ast.literal_eval(doc["choices"])


DOC_TO_TEXT = "{narrative}\n\n{question}\n\n{choices}\nAnswer:"


def doc_to_text(doc):
    """
    Convert a doc to text.
    """
    choices = ""
    for i, choice in enumerate(ast.literal_eval(doc["choices"])):
        choices += f"{i + 1} - {choice}\n"

    text = DOC_TO_TEXT.format(
        narrative=doc["narrative"], question=doc["question"], choices=choices
    )

    return text
