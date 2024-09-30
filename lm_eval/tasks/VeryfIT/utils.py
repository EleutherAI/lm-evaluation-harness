def doc_to_text(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    statement = doc["statement"]
    statement_date = doc["statement_date"]
    return f"Il seguente statement nella data indicata era vero o falso? Rispondi solo 'Vero' o 'Falso'.\nStatement: {statement}\nData: {statement_date}"

def doc_to_text_clarified(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    if doc["annotato"] and doc["statement_revised"] != "":
      statement = doc["statement_revised"]
    else:
      statement = doc["statement"]
    statement_date = doc["statement_date"]
    return f"Il seguente statement nella data indicata era vero o falso? Rispondi solo 'Vero' o 'Falso'.\nStatement: {statement}\nData: {statement_date}"