import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        ctx = (
            doc["norm"].capitalize()
            + " "
            + doc["situation"].capitalize()
            + " "
            + doc["intention"].capitalize()
        )
        choices = [doc["moral_action"], doc["immoral_action"]]
        out_doc = {
            "query": ctx,
            "choices": choices,
            "label": 0,
        }
        return out_doc

    return dataset.map(_process_doc)
