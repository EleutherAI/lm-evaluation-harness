def doc_to_text(doc):
    text = f"هدايت: {doc['Instruction']}\n"
    if doc.get("Input"):
        text += f"معلومات: {doc['Input']}\n"
    return text + "جواب:"
