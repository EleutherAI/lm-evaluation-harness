import sklearn
import numpy as np
import textdistance

orig_label_classes = ["1", "2", "3", "4", "5", "6", "7", "8"]
trans_label_classes = ['Identificação', 'Fatos', 'Argumentos', 'Fundamentação', 'Jurisprudência', 'Pedidos', 'Valor da Causa', 'Outros']

def doc_to_target(doc):
    return trans_label_classes[doc['label']]

def levenshtein_similarity_match(predictions, references):  # This is a passthrough function
    print(predictions, references)
    string_label = [s.lower() for s in trans_label_classes]
    
    prediction = predictions[0].strip().lower()
    get_text_until = ['.', ',', ';', '\n']
    for split_char in get_text_until:
        if split_char in prediction:
            prediction = prediction.split(split_char)[0]

    reference = references[0].lower()

    similarities = []
    for label in string_label:
        similarities.append(textdistance.levenshtein.normalized_similarity(prediction, label))
    if max(similarities) < 0.3:
        prediction = 'INVALID'
    else:
        prediction = string_label[similarities.index(max(similarities))]
    return (prediction, reference)

def agg_f1_macro(items):
    predictions, references = zip(*items)

    return sklearn.metrics.f1_score(y_true=references, y_pred=predictions, average='macro')

def agg_acc(items):
    predictions, references = zip(*items)

    return sklearn.metrics.accuracy_score(y_true=references, y_pred=predictions)