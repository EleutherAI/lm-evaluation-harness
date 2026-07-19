orig_label_classes = ["1", "2", "3", "4", "5", "6", "7", "8"]
trans_label_classes = ['Identificação das Partes', 'Fatos', 'Argumentos', 'Fundamentação Legal', 'Jurisprudência', 'Pedidos', 'Valor da Causa', 'Outros']

def doc_to_target(doc):
    return trans_label_classes[doc['label']]
