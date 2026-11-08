def p1(doc):
    candidates, choices = doc["choices"]["text"], doc["choices"]["label"]
    formatted_choices = "\n".join(
        [f"{choices[i]}: {candidates[i]}" for i, _ in enumerate(candidates)]
    )
    return f"{doc['question']}\n{formatted_choices}\nResposta:"


def p2(doc):
    candidates, choices = doc["choices"]["text"], doc["choices"]["label"]
    formatted_choices = "\n".join(
        [f"Opció {choices[i]}: {candidates[i]}" for i, _ in enumerate(candidates)]
    )
    return f"{doc['question']}\n{formatted_choices}\nÉs la resposta correcta {', '.join(choices[:-1])} o {choices[-1]}?"


def doc_to_choice(doc) -> list[str]:
    """
    Returns the answer choices for a document.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        list: A list of strings containing the answer choices.
    """
    letters = doc["choices"]["label"]
    return [f"Opció {letter}" for letter in letters]
