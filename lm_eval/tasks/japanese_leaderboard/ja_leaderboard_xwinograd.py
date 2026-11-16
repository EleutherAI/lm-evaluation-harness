def process_docs(dataset):
    def _add_choices_and_label(doc):
        doc["label"] = int(doc["answer"]) - 1
        doc["choices"] = [doc["sentence1"].strip(), doc["sentence2"].strip()]
        return doc

    return dataset.map(_add_choices_and_label)
