def doc_to_text(doc) -> str:
    SENTENCE1 = doc['Sentence1']
    SENTENCE2 = doc['Sentence2']
    doc_to_text = f"Data questa coppia di frasi, valuta se la prima frase implica la seconda. Rispondi solo 'sì' o 'no'.\n{SENTENCE1}\n{SENTENCE2}"
    doc_to_text += "\nRisposta:"

    return doc_to_text