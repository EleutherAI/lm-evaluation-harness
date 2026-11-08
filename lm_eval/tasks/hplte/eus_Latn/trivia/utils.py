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
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Galdera: {doc['question']}\n{formatted_choices}\nErantzuna:"


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
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}. {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"{doc['question']}\n{formatted_choices}\nErantzuna:"


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
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    choices = LETTERS[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice} aukera: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"{doc['question']}\n{formatted_choices}\n\nZein da erantzun zuzena, A, B, C ala D?"


def doc_to_choice_p01(doc) -> list[str]:
    """
    Returns the answer choices for a document.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        list: A list of strings containing the answer choices.
    """
    num_choices = len(doc["candidates"])
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
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
    if num_choices < 2:
        raise ValueError("Invalid number of candidates")
    return [f"{letter} aukera" for letter in LETTERS[:num_choices]]
