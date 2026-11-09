from typing import List


letters = ["A", "B", "C", "D"]


def doc_to_text(doc) -> str:
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
    choices = letters[:num_choices]
    formatted_choices = "\n".join(
        [f"{choice}: {candidates[i]}" for i, choice in enumerate(choices)]
    )
    return f"Galdera: {doc['question']}\n{formatted_choices}\nErantzuna:"


def doc_to_choice(doc) -> List[str]:
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
    return letters[:num_choices]
