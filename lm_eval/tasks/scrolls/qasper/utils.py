import transformers.data.metrics.squad_metrics as squad_metrics

def process_docs_bool(dataset):

    dataset = process_docs_prepended_question(dataset)

    return dataset.filter(lambda doc: squad_metrics.normalize_answer(doc["output"]) in ["yes", "no"])

def process_docs_freeform(dataset):

    dataset = process_docs_prepended_question(dataset)

    return dataset.filter(lambda doc: squad_metrics.normalize_answer(doc["output"]) not in ["yes", "no"])

def f1(prediction, reference):
    return squad_metrics.compute_f1(prediction[0], reference[0])
