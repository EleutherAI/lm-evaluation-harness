from collections.abc import Sequence

import numpy as np

from lm_eval.api.registry import register_reduction


@register_reduction("pass@k")
def pass_at_k(targets: Sequence[int], predictions: Sequence[int], k: int = 1) -> float:
    """
    From Chen et al. 2021: https://arxiv.org/abs/2107.03374
    n: total number of samples
    c: number of correct samples
    :param items: list of 0/1 predictions
    :param k: k in pass@k
    """
    assert len(targets) == len(predictions), (
        "Length of predictions and items must match."
    )
    n = len(predictions)
    c = len([1 for x, y in zip(predictions, targets, strict=True) if x == y])
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
