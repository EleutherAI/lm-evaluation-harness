upper_pronouns = [
    "A",
    "An",
    "The",
    "She",
    "He",
    "It",
    "They",
    "My",
    "His",
    "Her",
    "Their",
]


def process_doc(dataset):
    def process_fn(doc):
        # The HF implementation of `wsc273` is not `partial evaluation` friendly.
        doc["text"] = doc["text"].replace("  ", " ")
        doc["options"][0] = __normalize_option(doc, doc["options"][0])
        doc["options"][1] = __normalize_option(doc, doc["options"][1])
        return doc

    return dataset.map(process_fn)


def __normalize_option(doc, option):
    # Append `'s` to possessive determiner based options.
    if doc["pronoun"].lower() in ["my", "his", "her", "our", "their"]:
        option += "'s"
    # Appropriately lowercase the pronoun in the option.
    pronoun = option.split()[0]
    start_of_sentence = doc["text"][doc["pronoun_loc"] - 2] == "."
    if not start_of_sentence and pronoun in upper_pronouns:
        return option.replace(pronoun, pronoun.lower())
    return option
