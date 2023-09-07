import transformers.data.metrics.squad_metrics as squad_metrics

def process_docs(dataset):

    dataset = process_docs_prepended_question(dataset)

    def _process_doc(doc):

        doc["is_yes_no"] = reduce(lambda prev, cur: prev and squad_metrics.normalize_answer(cur)
                                  in ["yes", "no"], doc["outputs"], True)

        return doc

    return dataset.map(_process_doc)

def process_results(doc, results):
    if doc["is_yes_no"]:
        prediction = " yes" if results[0] > results[1] else " no"
    elif len(results[0].strip()) == 0:
        prediction = "Unanswerable"
    else:
        prediction = results[0]
    return {
        "f1": (prediction, doc["outputs"])
    }