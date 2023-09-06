def doc_to_text(doc):
    return f"Question: {doc['question']}\n(A) {doc['opa']}\n(B) {doc['opb']}\n(C) {doc['opc']}\n(D) {doc['opd']}\nAnswer: ("

def doc_to_target(doc):
    return '' + chr(ord('A') + int(doc["cop"]))