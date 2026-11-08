def paws_process_docs(dataset):
    empty_docs = []

    def _process_doc(doc):
        if doc["sentence1"] not in [None, ""] and doc["sentence2"] not in [None, ""]:
            # Remove final punctuation mark in the first sentence
            if doc["sentence1"].endswith((".", ",", ";")):
                doc["sentence1"] = doc["sentence1"][:-1]
            # Start the second sentence in lowercase (to be used after "Yes, ...")
            doc["sentence2"] = lowercase_first_letter(doc["sentence2"])
            return doc
        else:
            empty_docs.append(doc)
            return doc

    def lowercase_first_letter(text):
        return text[0].lower() + text[1:]

    return dataset.filter(
        lambda doc: (
            doc["sentence1"] not in [None, ""] and doc["sentence2"] not in [None, ""]
        )
    ).map(_process_doc)
