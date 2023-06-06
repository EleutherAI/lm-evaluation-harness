import math
from collections.abc import Iterable

import numpy as np
import sacrebleu
import sklearn.metrics
import random

from lm_eval.api.register import (
    register_metric,
    register_higher_is_better,
    register_output_type,
    register_default_aggregation,
)


@register_default_aggregation("mean")
@register_output_type("loglikelihood")
@register_output_type("multiple_choice")
@register_higher_is_better(True)
@register_metric("acc")
def acc_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("mean")
@register_output_type("multiple_choice")
@register_higher_is_better(True)
@register_metric("acc_norm")
def acc_norm_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("mean")
@register_output_type("multiple_choice")
@register_higher_is_better(True)
@register_metric("acc_mutual_info")
def acc_mutual_info_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("perplexity")
@register_output_type("loglikelihood")
@register_higher_is_better(False)
@register_metric("perplexity")
def perplexity_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("weighted_perplexity")
@register_output_type("loglikelihood_rolling")
@register_higher_is_better(False)
@register_metric("word_perplexity")
def word_perplexity_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("weighted_perplexity")
@register_output_type("loglikelihood_rolling")
@register_higher_is_better(False)
@register_metric("byte_perplexity")
def byte_perplexity_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("bits_per_byte")
@register_output_type("loglikelihood_rolling")
@register_higher_is_better(False)
@register_metric("bits_per_byte")
def bits_per_byte_fn(items):  # This is a passthrough function
    return items


@register_default_aggregation("mean")
@register_output_type("loglikelihood")
@register_higher_is_better(True)
@register_metric("acc_all")
def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


@register_default_aggregation("mean")
@register_higher_is_better(True)
@register_metric("matthews_corrcoef")
def matthews_corrcoef(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


@register_default_aggregation("mean")
@register_higher_is_better(True)
@register_metric("f1")
def f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)

    return np.max(fscore)


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


@register_default_aggregation("mean")
@register_higher_is_better(True)
@register_metric("bleu")
def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


@register_default_aggregation("mean")
@register_higher_is_better(True)
@register_metric("chrf")
def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """

    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


@register_default_aggregation("mean")
@register_higher_is_better(False)
@register_metric("ter")
def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score
