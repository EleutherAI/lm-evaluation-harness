import re
import string

import numpy as np


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def process_docs(dataset):
    def _process(doc):
        return {
            "id": doc["query_id"],
            "passage": doc["passage"],
            "question": doc["question"],
            "answers": get_answers(doc),
        }

    return dataset.map(_process)


def get_answers(doc):
    def _flatten_validated_answers(validated_answers):
        """Flattens a dict of lists of validated answers.
        {"number": ['1', '8'], ...}
        -> [{"number": ['1'], ...}, {"number": ['8'], ...}]
        """
        valid_answers = []
        for i in range(len(validated_answers["number"])):
            valid_answers.append(
                {
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                }
            )
        return valid_answers

    answers = []
    answers_set = set()
    candidates = [doc["answer"]] + _flatten_validated_answers(doc["validated_answers"])
    for candidate in candidates:
        answer = parse_answer(candidate)
        if answer in answers_set:
            continue
        answers_set.add(answer)
        answers.append(answer)
    return answers


def parse_answer(answer):
    # NOTE: Everything is returned as a tuple for uniformity and hashability.
    if answer["number"] != "":
        return (str(answer["number"]),)
    if answer["spans"] != []:
        return tuple(answer["spans"])
    return (
        " ".join(
            [answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]]
        ).strip(),
    )


def process_results(doc, results):
    preds, golds = results, doc["answers"]
    max_em = 0
    max_f1 = 0
    for gold_answer in golds:
        exact_match, f1_score = get_metrics(preds, gold_answer)
        if gold_answer[0].strip():
            max_em = max(max_em, exact_match)
            max_f1 = max(max_f1, f1_score)
    return {"em": max_em, "f1": max_f1}


def get_metrics(predicted, gold):
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(
        gold_bags[0]
    ):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def _answer_to_bags(answer):
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = _normalize(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
    return normalized_spans, token_bags


def _align_bags(predicted, gold):
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    from scipy.optimize import linear_sum_assignment

    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, pred_item):
                scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag, gold_bag):
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if not (precision == 0.0 and recall == 0.0)
        else 0.0
    )
    return f1


def _match_numbers_if_present(gold_bag, predicted_bag):
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def _remove_articles(text):
    return _ARTICLES.sub(" ", text)


def _white_space_fix(text):
    return " ".join(text.split())


def _remove_punc(text):
    exclude = set(string.punctuation)
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in exclude)
    else:
        return text


def _fix_number(text):
    return str(float(text)) if _is_number(text) else text


def _tokenize(text):
    return re.split(" |-", text)


def _normalize(answer):
    tokens = [
        _white_space_fix(_remove_articles(_fix_number(_remove_punc(token.lower()))))
        for token in _tokenize(answer)
    ]
    tokens = [token for token in tokens if token.strip()]
    normalized = " ".join(tokens).strip()
    return normalized
