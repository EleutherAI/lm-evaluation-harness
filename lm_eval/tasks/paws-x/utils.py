import re


def general_detokenize(string):
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def lowercase_first_letter(text):
    return text[0].lower() + text[1:]


def process_docs_paraphrases(dataset):
    empty_docs = []

    def _process_doc(doc):
        if doc["sentence1"] not in [None, ""] and doc["sentence2"] not in [None, ""]:
            doc["sentence1"] = general_detokenize(doc["sentence1"]).strip()
            doc["sentence2"] = general_detokenize(doc["sentence2"]).strip()
            # Remove final punctuation mark in the first sentence
            if doc["sentence1"].endswith((".", ",", ";")):
                doc["sentence1"] = doc["sentence1"][:-1]
            # Start the second sentence in lowercase (to be used after "Yes, ...")
            doc["sentence2"] = lowercase_first_letter(doc["sentence2"])
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
        lambda doc: doc["sentence1"] not in [None, ""]
        and doc["sentence2"] not in [None, ""]
    ).map(_process_doc)
