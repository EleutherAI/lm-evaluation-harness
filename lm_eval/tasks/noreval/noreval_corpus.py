"""Corpus-level metrics shared by the generation tasks.

Each document contributes K sampled generations (one per repetition). The
point estimate treats repetition k as its own parallel corpus, made of every
document's k-th sample, computes the corpus score on each of the K corpora
separately, and averages the K scores. The `*_stderr` functions bootstrap that
average by resampling documents with replacement (seeded, so results are
reproducible), each document's K samples moving together.
"""

import logging
import random
import statistics

import numpy as np
import sklearn.metrics
from sacrebleu.metrics import BLEU, CHRF


logger = logging.getLogger(__name__)

BOOTSTRAP_ITERS = 1000
BOOTSTRAP_SEED = 1234


def _repetitions(items):
    """Regroup the per-document sample lists into one corpus per repetition.

    Corpus k holds the k-th (reference, prediction) pair of every document,
    together with the document's index so the bootstrap can resample by
    document. Documents with fewer than k+1 samples are absent from corpus k.
    """
    n_reps = max(len(doc_pairs) for doc_pairs in items)
    return [
        [(i, doc_pairs[k]) for i, doc_pairs in enumerate(items) if k < len(doc_pairs)]
        for k in range(n_reps)
    ]


def _corpus_score(metric, pairs):
    return metric.corpus_score(
        [prediction for _, prediction in pairs],
        [[reference for reference, _ in pairs]],
    ).score


def _mean_corpus_score(metric, items):
    return statistics.fmean(
        _corpus_score(metric, [pair for _, pair in corpus])
        for corpus in _repetitions(items)
    )


def mean_bleu(items):
    return _mean_corpus_score(BLEU(), items)


def mean_chrf(items):
    return _mean_corpus_score(CHRF(), items)


def bleu_stderr(items):
    return _bootstrap_stderr(items, BLEU(), mean_bleu(items))


def chrf_stderr(items):
    return _bootstrap_stderr(items, CHRF(), mean_chrf(items))


def _bootstrap_stderr(items, metric, direct_score):
    """Bootstrap the repetition-averaged corpus score by resampling documents.

    Uses sacrebleu's per-sentence statistics, verified against the directly
    computed score, so each replicate is a cheap re-aggregation: for every
    repetition the sampled documents' rows are re-aggregated into a corpus
    score, and the K corpus scores are averaged.
    """
    try:
        rows_by_rep = []
        for corpus in _repetitions(items):
            rows = metric._extract_corpus_statistics(
                [prediction for _, (_, prediction) in corpus],
                [[reference for _, (reference, _) in corpus]],
            )
            rows_by_rep.append(dict(zip((i for i, _ in corpus), rows, strict=True)))

        reproduced = statistics.fmean(
            metric._aggregate_and_compute(list(rows.values())).score
            for rows in rows_by_rep
        )
        if abs(reproduced - direct_score) > 1e-6:
            raise ValueError("statistics do not reproduce the corpus score")

        rng = random.Random(BOOTSTRAP_SEED)
        n = len(items)
        replicates = []
        for _ in range(BOOTSTRAP_ITERS):
            sample = [rng.randrange(n) for _ in range(n)]
            scores = []
            for rows in rows_by_rep:
                sampled_rows = [rows[i] for i in sample if i in rows]
                if sampled_rows:
                    scores.append(metric._aggregate_and_compute(sampled_rows).score)
            replicates.append(statistics.fmean(scores))
        return statistics.stdev(replicates)
    except Exception as exc:  # noqa: BLE001
        logger.error("bootstrap stderr failed (%s): %s", type(metric).__name__, exc)
        return float("nan")


def multi_f1(items):
    """Macro-average F1 over (prediction, gold) pairs.

    zero_division=0 makes degenerate bootstrap replicates (all predictions in
    one class) silent; sklearn's default computes the same value but warns.
    """
    preds, golds = zip(*items, strict=True)
    return sklearn.metrics.f1_score(
        np.array(golds), np.array(preds), average="macro", zero_division=0
    )


def multi_f1_stderr(items):
    """Bootstrap standard error of the macro-F1: the (prediction, gold) pairs
    are resampled with replacement and `multi_f1` is recomputed on each
    replicate.
    """
    rng = random.Random(BOOTSTRAP_SEED)
    pairs = list(items)
    n = len(pairs)
    replicates = [
        multi_f1([pairs[rng.randrange(n)] for _ in range(n)])
        for _ in range(BOOTSTRAP_ITERS)
    ]
    return statistics.stdev(replicates)
