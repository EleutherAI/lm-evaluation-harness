import math
from typing import Any

import numpy as np

from lm_eval.api.registry import register_aggregation as aggregation


@aggregation("bypass")
def bypass_agg(arr: Any):
    return 999


@aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@aggregation("median")
def median(arr):
    return sorted(arr)[len(arr) // 2]


@aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def weighted_mean(items):
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)
