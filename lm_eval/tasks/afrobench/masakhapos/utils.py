from lm_eval.utils import weighted_f1_score


def doc_to_text(doc):
    output = """Please provide the POS tags for each word in the input sentence. The input will be a list of words in
    the sentence. The output format should be a list of tuples, where each tuple consists of a word from the input text
    and its corresponding POS tag label from the tag label set: ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT" "SCONJ", "SYM", "VERB", "X"]. \nYour response should include only a
    list of tuples, in the order that the words appear in the input sentence, with each tuple containing the
    corresponding POS tag label for a word.

    Input: {tokens}
    Output: """

    text = output.format(subject=doc["tokens"])
    return text


def doc_to_target(doc):
    pos_tag_map = {
        0: "NOUN",
        1: "PUNCT",
        2: "ADP",
        3: "NUM",
        4: "SYM",
        5: "SCONJ",
        6: "ADJ",
        7: "PART",
        8: "DET",
        9: "CCONJ",
        10: "PROPN",
        11: "PRON",
        12: "X",
        13: "_",
        14: "ADV",
        15: "INTJ",
        16: "VERB",
        17: "AUX",
    }
    return [pos_tag_map[tag] for tag in doc["upos"]]
