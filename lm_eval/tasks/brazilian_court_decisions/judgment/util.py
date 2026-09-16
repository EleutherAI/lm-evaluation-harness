orig_label_classes = ["no", "partial", "yes"]
trans_label_classes = ["Negada", "Parcial", "Aceita"]

def doc_to_target(doc):
    return trans_label_classes[doc['label']]
