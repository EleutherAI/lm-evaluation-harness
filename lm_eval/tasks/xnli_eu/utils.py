from lm_eval.utils import general_detokenize

def lowercase_first_letter(text):
    return text[0].lower() + text[1:]

def uppercase_first_letter(text):
    return text[0].upper() + text[1:]

def process_doc_nli(dataset):

    def filter_fn(doc):
        # There shouldn't be any final punctuation marks (except periods) in the premise or the hypothesis.
        # They're supposed to be one single sentence in order to be concatenated properly in the prompt.
        if any([punct in sent for punct in ["¡", "!", "?", "¿", "...", ":", ";"] for sent in [doc["premise"], doc["hypothesis"]]]):
            return False

        return True

    def process_fn(doc):
        # Detokenize(remove extra whitespaces)
        doc["premise"] = general_detokenize(doc["premise"]).strip()
        doc["hypothesis"] = general_detokenize(doc["hypothesis"]).strip()

        # Remove periods from the end of the premise
        doc["premise"] = doc["premise"].rstrip(".")

        # Lowercase the first letter in the hypothesis
        doc["hypothesis"] = lowercase_first_letter(doc["hypothesis"])

        # Uppercase the first letter in the premise
        doc["premise"] = uppercase_first_letter(doc["premise"])

        # Ensure that the hypothesis ends with a single period
        doc["hypothesis"] = doc["hypothesis"].rstrip(".") + "."

        return doc

    return dataset.filter(filter_fn).map(process_fn)
