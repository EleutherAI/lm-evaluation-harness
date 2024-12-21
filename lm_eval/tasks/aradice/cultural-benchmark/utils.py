def process_docs(dataset):
    def _helper(doc):
        doc["choices"] = [doc["Option A"], doc["Option B"], doc["Option C"]]
        return doc

    return dataset.map(_helper)
