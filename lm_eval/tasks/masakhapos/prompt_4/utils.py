from lm_eval.utils import weighted_f1_score


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
        17: "AUX"
    }
    return [pos_tag_map[tag] for tag in doc["upos"]]
