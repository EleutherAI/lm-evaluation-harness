def doc_to_text(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N']
    question = doc['question']
    doc_to_text = f"Di seguito è riportata una domanda a scelta multipla e le possibili risposte. Scegli la lettera che meglio risponde alla domanda. Mantieni la tua risposta il più breve possibile; indica solo la lettera corrispondente alla tua risposta senza spiegazioni."
    doc_to_text += f"\nDomanda: {question}"
    doc_to_text += f"\nPossibili risposte:"

    for i in range(len(doc["choices"])):
      doc_to_text += f"\n{letters[i]}) {doc['choices'][i]}"
    doc_to_text += "\nRisposta:"

    return doc_to_text

def doc_to_choice(doc) -> str:
    """
    Converts a document to a formatted string.

    Args:
        doc (dict): A dictionary containing the document information.

    Returns:
        str: A formatted string containing the question and answer choices.
    """
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N']
    return letters[:len(doc['choices'])]
def macro_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore

