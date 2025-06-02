from itertools import chain

from sklearn.metrics import accuracy_score

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
        17: "AUX",
    }
    return [pos_tag_map[tag] for tag in doc["upos"]]


def acc_score(items):
    unzipped_list = list(zip(*items))

    golds, preds = unzipped_list[0], unzipped_list[1]

    # Flatten preds' inner lists
    flattened_preds = [list(chain.from_iterable(p)) for p in preds]

    # Calculate the accuracy for each gold-pred pair
    accuracy_scores = []
    for gold, pred in zip(golds, flattened_preds):
        # Ensure both lists are of the same length, otherwise truncate to match
        min_length = min(len(gold), len(pred))
        gold = gold[:min_length]
        pred = pred[:min_length]

        # Calculate accuracy for the current pair and add to the list
        accuracy = accuracy_score(gold, pred)
        accuracy_scores.append(accuracy)

    mean_accuracy = (
        sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    )
    return mean_accuracy
