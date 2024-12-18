def doc_to_text(doc) -> str:
    question = doc['Question Text']
    doc_to_text = f"Domanda: {question}\n"
    doc_to_text += "Opzioni: 'A. Vero' , 'B. Falso'\nIstruzioni: Devi restituire solo la lettera corrispondente alla risposta esatta.\nFormato della risposta: 'lettera'\nRisposta:"

    return doc_to_text

def doc_to_target(doc) -> str:
    if doc['Answer Type']:
      return 'A'
    else:
      return 'B'