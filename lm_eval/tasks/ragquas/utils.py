def doc_to_text(doc):
    context = "\n".join(
        [f"{k.replace('_','o ')}:\n{v}" for k, v in doc.items() if "context" in k and v!=""]
    )
    question = doc["question"]
    doc_to_text = f"""Responde a la pregunta utilizando únicamente los contextos proporcionados.

    Contextos:

    {context}

    Pregunta: {question}

    Respuesta:"""
    return doc_to_text
