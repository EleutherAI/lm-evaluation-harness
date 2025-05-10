import re
import string
from collections import Counter


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1(items):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]

    f1_list = []

    for i in range(len(golds)):
        prediction_tokens = normalize_answer(preds[i]).split()
        references_tokens = normalize_answer(golds[i]).split()
        common = Counter(prediction_tokens) & Counter(references_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1_score = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(references_tokens)
            f1_score = (2 * precision * recall) / (precision + recall)

        f1_list.append(f1_score)

    return sum(f1_list) / len(f1_list)
