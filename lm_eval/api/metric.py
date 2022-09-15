import logging
import math
import random
import numpy as np
import sacrebleu
import sklearn.metrics
from collections.abc import Iterable
from rouge_score import rouge_scorer
from typing import List, Mapping, Optional

from lm_eval.metrics import sari as sari_impl


logger = logging.getLogger(__name__)


def mean(arr):
    return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    if len(arr) == 1:
        return 0
    else:
        return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def median(arr):
    return arr[len(arr) // 2]


def matthews_corrcoef(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


def f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)
    return np.max(fscore)


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


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def compute_parity_scores(items):
    # Parity checks whether predictions in subsequent pairs of examples are consistent.
    # In WinogenderSchema those examples differ only in the gender of the pronoun in the hypothesis.
    indices2predictions = {idx: pred for idx, pred in items}
    parity_scores = []
    for idx in indices2predictions.keys():
        if (idx % 2) == 0 and (idx + 1) in indices2predictions:
            parity_scores.append(
                int(indices2predictions[idx] == indices2predictions[idx + 1])
            )
    return parity_scores


def parity(items):
    parity_scores = compute_parity_scores(items)
    if len(parity_scores) > 0:
        acc = mean(parity_scores)
    else:
        acc = 0.0
    return acc


def parity_stderr(items):
    parity_scores = compute_parity_scores(items)
    if len(parity_scores) > 0:
        stderr = mean_stderr(parity_scores)
    else:
        stderr = 0.0
    return stderr


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def perplexity(items):
    return math.exp(-mean(items))


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def sari(sentence_to_simplifiy, generated_sentence, references):
    """Implementation of SARI from the authors'."""
    return sari_impl.SARIsent(sentence_to_simplifiy, generated_sentence, references)


def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or uni-gram would be each token and a bi-gram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


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


def rouge(
    refs: List[str],
    pred: str,
    rouge_types: Optional[List[str]] = ["rouge1", "rouge2", "rougeL", "rougeLsum"],
) -> Mapping[str, float]:
    """ROUGE with multi-reference support

    Implementation based on GEM-metrics:
    https://github.com/GEM-benchmark/GEM-metrics/blob/431a8174bd6b3637e8d6118bfad2983e39e99733/gem_metrics/rouge.py

    Args:
        refs (List[str]):
            A `list` of reference `str`s.
        pred (str):
            A single prediction `str`s.
        rouge_types (Optional[List[str]]):
            An optional list of ROUGE types from the set:
            ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    Returns:
        A `dict` of ROUGE scores.
    """

    # Add newlines between sentences to correctly compute `rougeLsum`.
    if "rougeLsum" in rouge_types:
        # TODO: Adapt this to handle languages that do not support sentence endings by `.`.
        # See GEM-metrics implementation with lang specific `nltk` tokenizers to
        # split sentences.
        pred = pred.replace(".", ".\n")
        refs = [ref.replace(".", ".\n") for ref in refs]

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=True)
    # ROUGE multi-ref jackknifing
    if len(refs) > 1:
        cur_scores = [scorer.score(ref, pred) for ref in refs]

        # get best score for all leave-one-out sets
        best_scores = []
        for leave in range(len(refs)):
            cur_scores_leave_one = [
                cur_scores[s] for s in range(len(refs)) if s != leave
            ]
            best_scores.append(
                {
                    rouge_type: max(
                        [s[rouge_type] for s in cur_scores_leave_one],
                        key=lambda s: s.fmeasure,
                    )
                    for rouge_type in rouge_types
                }
            )
        # average the leave-one-out bests to produce the final score
        score = {
            rouge_type: rouge_scorer.scoring.Score(
                np.mean([b[rouge_type].precision for b in best_scores]),
                np.mean([b[rouge_type].recall for b in best_scores]),
                np.mean([b[rouge_type].fmeasure for b in best_scores]),
            )
            for rouge_type in rouge_types
        }
    else:
        score = scorer.score(refs[0], pred)
    # convert the named tuples to plain nested dicts
    score = {
        rouge_type: {
            "precision": score[rouge_type].precision,
            "recall": score[rouge_type].recall,
            "fmeasure": score[rouge_type].fmeasure,
        }
        for rouge_type in rouge_types
    }
    return score


# Standard Error Utils


class _BootstrapInternal:
    def __init__(self, f, n):
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - I considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are usually pretty big.
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    logger.info("Bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _BootstrapInternal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr, parity: parity_stderr}

    return stderr.get(metric, None)
