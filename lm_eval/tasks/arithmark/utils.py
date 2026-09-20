import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "ctx": doc["ctx"],
            "endings": doc["endings"],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def process_docs_easy(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "ctx": doc["ctx"],
            "endings": doc["endings"],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.filter(lambda doc: doc["metadata"]["difficulty"] == "easy").map(_process_doc)


def process_docs_medium(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "ctx": doc["ctx"],
            "endings": doc["endings"],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.filter(lambda doc: doc["metadata"]["difficulty"] == "medium").map(_process_doc)


def process_docs_hard(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "ctx": doc["ctx"],
            "endings": doc["endings"],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.filter(lambda doc: doc["metadata"]["difficulty"] == "hard").map(_process_doc)
