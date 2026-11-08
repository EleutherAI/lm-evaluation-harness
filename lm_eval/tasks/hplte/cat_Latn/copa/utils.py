def lowercase_first_letter(text):
    return text[0].lower() + text[1:]


def process_docs_copa_ca(dataset):
    def _process_doc(doc):
        doc["choice1"] = lowercase_first_letter(doc["choice1"])
        doc["choice2"] = lowercase_first_letter(doc["choice2"])
        return doc

    return dataset.map(_process_doc)
