def process_docs(dataset):
    def _add_choices(doc):
        doc["choices"] = [doc[f"choice{i}"] for i in range(5)]
        return doc

    return dataset.map(_add_choices)
