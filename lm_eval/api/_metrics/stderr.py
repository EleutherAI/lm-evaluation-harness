import math
import os
import random
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from lm_eval.api._metrics.aggregations import mean, median, nanmean, perplexity


T = TypeVar("T")


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr: Sequence[T]) -> float:
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


class _bootstrap_internal:
    """
    Pool worker: `(i, xs)` → `n` bootstrap replicates
    of `f(xs)`using a RNG seeded with `i`.
    """

    def __init__(self, f: Callable[[Sequence[T]], float], n: int) -> None:
        self.f = f
        self.n = n

    def __call__(self, v: tuple[int, Sequence[T]]) -> list[float]:
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def _bootstrap_internal_no_mp(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> list[float]:
    """
    Single-process fallback: compute `iters` bootstrap replicates
    of statistic`f(xs)`, chunked (≤ 1000 draws).
    """
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print(f"Bootstrapping for stddev: {getattr(f, '__name__', repr(f))}")

    # A single loop replaces the multiprocessing pool.
    for i in tqdm(range(iters // chunk_size)):
        rnd = random.Random(i)
        for _ in range(chunk_size):
            res.append(f(rnd.choices(xs, k=len(xs))))

    return res


def bootstrap_stderr(
    f: Callable[[Sequence[T]], float], xs: Sequence[T], iters: int
) -> float:
    """
    Bootstrap estimate of the standard error of statistic `f(xs)`
    using up to `iters` resamples, chunked (≤ 1000 draws)

    Executes in parallel unless the env-var `DISABLE_MULTIPROC` is set;
    """
    if not os.getenv("DISABLE_MULTIPROC"):
        import multiprocessing as mp

        # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
        # equivalent to stderr calculated without Bessel's correction in the stddev.
        # Unfortunately, I haven't been able to figure out what the right correction is
        # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
        # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
        # Thankfully, shouldn't matter because our samples are pretty big usually anyways
        res = []
        chunk_size = min(1000, iters)
        from tqdm import tqdm

        print(f"Bootstrapping for stddev: {getattr(f, '__name__', repr(f))}")
        with mp.Pool(mp.cpu_count()) as pool:
            for bootstrap in tqdm(
                pool.imap(
                    _bootstrap_internal(f, chunk_size),
                    [(i, xs) for i in range(iters // chunk_size)],
                ),
                total=iters // chunk_size,
            ):
                # sample w replacement
                res.extend(bootstrap)
    else:
        res = _bootstrap_internal_no_mp(f, xs, iters)

    return sample_stddev(res)


def stderr_for_metric(
    metric: Callable[[Sequence[T]], float], bootstrap_iters: int
) -> Callable[[Sequence[T]], float] | None:
    """
    Return a function that estimates the standard error of `metric(xs)`.

    * If `bootstrap_iters > 0` and the metric is in the pre-approved
      bootstrappable list, use `bootstrap_stderr` with that many draws.
    * If the metric has a closed-form SE (e.g. `mean`, `acc_all`), use it.
    * Otherwise, return `None`.
    """

    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        median,
        perplexity,
        nanmean,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr}

    return stderr.get(metric)


def pooled_sample_stderr(stderrs: list[float], sizes: list[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum(
            [
                (size - 1) * stderr**2 * size
                for size, stderr in zip(sizes, stderrs, strict=True)
            ]
        )
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def combined_sample_stderr(stderrs: list[float], sizes: list[int], metrics=None):
    assert metrics is not None, (
        "Need to pass a list of each subtask's metric for this stderr aggregation"
    )
    assert len(stderrs) == len(sizes) and len(sizes) == len(metrics)

    # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1390 for more documentation.
    # This formula depends on sample means.
    # removed because it seems to give erroneously huge stderrs for groupings of tasks
    # and does not seem to match up with bootstrap-calculated stderrs for groups.

    ### don't use this unless a statistician has told you it's the right thing to do ###

    # accumulators: we'll aggregate pairwise N - 1 times
    variance = stderrs[0] ** 2
    curr_size = sizes[0]
    curr_score = metrics[0]

    for stderr, size, score in zip(stderrs[1:], sizes[1:], metrics[1:], strict=True):
        curr_score = ((curr_score * curr_size) + (score * size)) / (
            curr_size + size
        )  # NOTE: this assumes our aggregation fn is "mean"

        variance = ((curr_size - 1) * variance + (size - 1) * (stderr**2)) / (
            curr_size + size - 1
        ) + curr_size * size / ((curr_size + size) * (curr_size + size - 1)) * (
            curr_score - score
        ) ** 2

    return np.sqrt(variance)
