def p1(doc):
    candidates, letters = doc["choices"]["text"], doc["choices"]["label"]
    formatted_choices = "\n".join(
        [f"{letters[i]}: {candidates[i]}" for i, _ in enumerate(candidates)]
    )
    return f"{doc['question']}\n{formatted_choices}\nZein da erantzun zuzena, {', '.join(letters[:-1])} ala {letters[-1]}?"


def p2(doc):
    candidates, letters = doc["choices"]["text"], doc["choices"]["label"]
    formatted_choices = "\n".join(
        [f"{letters[i]}: {candidates[i]}" for i, _ in enumerate(candidates)]
    )
    return f"{doc['question']}\n{formatted_choices}\nErantzuna:"


def doc_to_choice(doc) -> list[str]:
    """
    Returns the answer choices for a document.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        list: A list of strings containing the answer choices.
    """
    letters = doc["choices"]["label"]
    return [f"Aukera {letter}" for letter in letters]
