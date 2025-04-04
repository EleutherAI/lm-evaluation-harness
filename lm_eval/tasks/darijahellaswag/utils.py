import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = doc["ctx"]
        out_doc = {
            "query": doc["activity_label"] + ": " + ctx,
            "choices": doc["endings"],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)
