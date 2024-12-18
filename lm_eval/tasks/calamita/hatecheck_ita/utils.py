import pdb

def _preprocess(doc):
    doc["hateful"] = 1 if doc["label_gold"] == "hateful" else 0
    return doc

def process_docs(dataset):
    return dataset.map(_preprocess)


def process_results(doc, predictions):
    response = 0 if predictions[0][1] == True else 1
    return (1 if doc["hateful"] == response else 0, doc["target_ident"])
