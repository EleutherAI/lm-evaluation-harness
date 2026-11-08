from lm_eval.utils import general_detokenize


def lowercase_first_letter(text):
    return text[0].lower() + text[1:]


def process_doc_nli(dataset):
    def process_fn(doc):
        # Detokenize(remove extra whitespaces)
        doc["premise"] = general_detokenize(doc["premise"]).strip()
        doc["hypothesis"] = general_detokenize(doc["hypothesis"]).strip()
        # Remove last punctuation mark in the premise
        doc["premise"] = (
            doc["premise"][:-1]
            if doc["premise"].endswith((".", ",", "!", "?"))
            else doc["premise"]
        )
        # Lowercase the first letter in the hypothesis
        doc["hypothesis"] = lowercase_first_letter(doc["hypothesis"])
        # Ensure that the hypothesis ends with a dot
        doc["hypothesis"] = (
            (doc["hypothesis"] + ".")
            if not doc["hypothesis"].endswith(".")
            else doc["hypothesis"]
        )
        return doc

    return dataset.map(process_fn)
