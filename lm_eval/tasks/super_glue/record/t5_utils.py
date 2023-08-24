import re
import string
import collections
import numpy as np

from tqdm import tqdm
from datasets import Dataset, concatenate_datasets

from lm_eval.api.metrics import metric_max_over_ground_truths


def doc_to_text(doc):

    passage = doc['passage']
    passage = re.sub(r'(\.|\?|\!|\"|\')\n@highlight\n', r'\1 ', passage)
    passage = re.sub(r'\n@highlight\n', '. ', passage)

    return " ".join([
                "record query:",
                doc['query'],
                "entities:",
                ", ".join(doc['entities']),
                "passage:",
                passage
            ])


def process_docs(dataset):

    def split_answers(doc):
        split_doc = {
            **{k: [] for k in doc.keys()},
        }
        answers = doc.pop("answers")
        for idx, answer in enumerate(answers):

            for key in split_doc.keys():
                if key in doc:
                    split_doc[key].append(doc[key])
                
            split_doc["answers"].append(answer)
        return split_doc
    
    dataset = dataset.map(split_answers)
    new_dataset = {}
    for key in dataset.features.keys():
        new_dataset[key] = [x for row in dataset[key] for x in row]
    
    return Dataset.from_dict(new_dataset)


def deduplicate_metric(metric_fn,
                       group_key: str = "group",
                       value_key: str = "value"):
    """Returns a metric that only considers one example per group.

    Useful for things like ReCoRD where inputs may be replicated during training
    to handle multiple labels, but where at eval we only want a single copy of
    each example.

    Args:
        metric_fn: function, the metric to compute on the unique examples.
        group_key: the key for the grouping value in the target dictionary.
        value_key: the key for the value in the dictionaries.

    Returns:
        A metric function that deduplicated based on the grouping key before
        returning a metric.
    """
    def _deduplicated_metric(targets, predictions):
        """Deduplicate targets and predictions and pass that to the metric fn."""
        processed_groups = set()
        deduplicated_targets = []
        deduplicated_predictions = []
        for targ, pred in zip(targets, predictions):
            group = targ[group_key]
            if group in processed_groups:
                continue
            processed_groups.add(group)
            deduplicated_targets.append(targ[value_key])
            deduplicated_predictions.append(pred[value_key])
        return metric_fn(deduplicated_targets, deduplicated_predictions)
    return _deduplicated_metric


def normalize_squad(answer):
    """Normalization used in official SQuAD evaluation script."""
    def _normalize_answer(text, punc_chars, punc_repl):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(s):
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def replace_punctuation(s):
            to_replace = set(punc_chars)
            return "".join(punc_repl if ch in to_replace else ch for ch in s)

        def white_space_fix(s):
            return " ".join(s.split())

        text = text.lower()
        text = replace_punctuation(text)
        text = remove_articles(text)
        text = white_space_fix(text)

        return text

    return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")

def em(predictions, references):  # This is a passthrough function
    return (predictions[0], references[0])

def f1(predictions, references):  # This is a passthrough function
    return (predictions[0], references[0])

def squad_em_agg(items):

    def _exact_match_score(target, prediction):
        return target == prediction

    grouped_values = collections.defaultdict(lambda: ([], []))
    for prediction, reference in items:
        group, reference = reference.split("_")
        grouped_values[group][0].append(normalize_squad(prediction))
        grouped_values[group][1].append(normalize_squad(reference))

    print(grouped_values)
    import sys; sys.exit()
    em = np.mean([
        metric_max_over_ground_truths(_exact_match_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    return em

def squad_f1_agg(items):

    def _f1_score(target, prediction):
        """Computes token f1 score for a single target and prediction."""
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        common = (collections.Counter(prediction_tokens) &
                    collections.Counter(target_tokens))
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(target_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    predictions, targets = zip(*items)
    targets = [normalize_squad(t) for t in targets]
    predictions = [normalize_squad(p) for p in predictions]

    f1 = np.mean([
        metric_max_over_ground_truths(_f1_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    return f1
