def doc_to_text(doc):
    """
    Formats the L-Eval input document and instruction.
    """
    context = doc.get("input", "")
    instructions = doc.get("instructions", [])
    
    if isinstance(instructions, list):
        instruction_str = "\n".join(instructions)
    else:
        instruction_str = str(instructions)

    return f"Document:\n{context}\n\nInstruction:\n{instruction_str}\n\nAnswer:"


def doc_to_target(doc):
    """
    Extracts the reference answer/output string from the document.
    """
    outputs = doc.get("outputs", [])
    if isinstance(outputs, list):
        return outputs[0] if len(outputs) > 0 else ""
    return str(outputs)