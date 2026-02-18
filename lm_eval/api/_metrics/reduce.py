from collections.abc import Sequence
from typing import Any

import numpy as np

from lm_eval.api.registry import register_reduction


@register_reduction("pass@k")
def pass_at_k(
    references: Any, predictions: Sequence[int] | Sequence[bool], *, k: int = 1
) -> float:
    """Estimate pass@k from Chen et al. 2021 (https://arxiv.org/abs/2107.03374).

    Predictions are per-repeat metric scores (e.g. 0/1 from exact_match).
    A truthy score counts as a pass.

    Args:
        references: Unused.
        predictions: Per-repeat metric scores for this document.
        k: k in pass@k.
    """
    n = len(predictions)
    c = sum(1 for p in predictions if p)
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def take_first(references, predictions: Sequence):
    """Return the first repeat's prediction, ignoring all others."""
    return predictions[0]
