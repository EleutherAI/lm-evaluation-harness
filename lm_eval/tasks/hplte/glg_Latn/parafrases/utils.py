from lm_eval.utils import general_detokenize


def lowercase_first_letter(text):
    return text[0].lower() + text[1:]


def process_docs_paraphrases(dataset):
    empty_docs = []

    def _process_doc(doc):
        if doc["Frase"] not in [None, ""] and doc["Paráfrase"] not in [None, ""]:
            doc["Frase"] = general_detokenize(doc["Frase"]).strip()
            doc["Paráfrase"] = general_detokenize(doc["Paráfrase"]).strip()
            # Remove final punctuation mark in the first sentence
            if doc["Frase"].endswith((".", ",", ";")):
                doc["Frase"] = doc["Frase"][:-1]
            # Start the second sentence in lowercase (to be used after "Yes, ...")
            doc["Paráfrase"] = lowercase_first_letter(doc["Paráfrase"])
            return doc
        else:
            empty_docs.append(doc)
            return doc

    if empty_docs != []:
        len_empty_docs = len(empty_docs)
        print(
            f"Found {len_empty_docs} empty documents out of the {len(dataset)} total docs in the dataset: {empty_docs}"
        )
    return dataset.filter(
        lambda doc: (
            doc["Frase"] not in [None, ""] and doc["Paráfrase"] not in [None, ""]
        )
    ).map(_process_doc)
