LETTERS = ["A", "B", "C", "D"]


def p0(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    candidates = doc["candidates"]
    num_choices = len(candidates)
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Pasartea: {doc['context']}\n\nGaldera: {doc['question']}\n{formatted_choices}\nErantzuna:"


def p1(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    candidates = doc["candidates"]
    num_choices = len(candidates)
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Testuingurua: {doc['context']}\n\nGaldera: {doc['question']}\n{formatted_choices}\nErantzuna:"


def p2(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    candidates = doc["candidates"]
    num_choices = len(candidates)
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"Auskera {choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Irakurri pasartea eta erantzun galderari: {doc['context']}\n{doc['question']}\n{formatted_choices}\nZein da erantzun zuzena {', '.join(choices)}?"


def doc_to_choice_p01(doc) -> list[str]:
    """
    Returns the answer choices for a document.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        list: A list of strings containing the answer choices.
    """
    num_choices = len(doc["candidates"])
    return LETTERS[:num_choices]


def doc_to_choice_p2(doc) -> list[str]:
    """
    Returns the answer choices for a document.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        list: A list of strings containing the answer choices.
    """
    num_choices = len(doc["candidates"])
    return [f"Aukera {letter}" for letter in LETTERS[:num_choices]]
