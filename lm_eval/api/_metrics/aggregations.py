import math

import numpy as np

from lm_eval.api.registry import register_aggregation


@register_aggregation("bypass")
def bypass_agg(arr):
    return 999


@register_aggregation("nanmean")
def nanmean(arr):
    if len(arr) == 0 or all(np.isnan(arr)):
        return np.nan
    return np.nanmean(arr)


@register_aggregation("mean")
def mean(arr):
    return sum(arr) / len(arr)


@register_aggregation("median")
def median(arr):
    return arr[len(arr) // 2]


@register_aggregation("perplexity")
def perplexity(items):
    return math.exp(-mean(items))


@register_aggregation("weighted_perplexity")
def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


@register_aggregation("bits_per_byte")
def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def weighted_mean(items):
    a, b = zip(*items, strict=True)
    return sum(a) / sum(b)
