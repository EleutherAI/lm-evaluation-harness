orig_label_classes = ['ARGUMENTACAO', 'DECISAO', 'DEFINICAO', 'EXPOSICAO', 'PEDIDO', 'QUALIFICACAO', 'REFERENCIA', 'SEM_CLASSE', 'VERBETACAO']
trans_label_classes = ['Argumentação', 'Decisão', 'Definição', 'Exposição', 'Pedido', 'Qualificação', 'Referência', 'Sem Classe', 'Vebertação']

def doc_to_target(doc):
    return trans_label_classes[doc['label']]
