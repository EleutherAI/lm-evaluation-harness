from collections.abc import Sequence

import numpy as np

from lm_eval.api.registry import register_reduction


def _expand_references(references, predictions: Sequence) -> Sequence:
    """Expand a scalar reference to a parallel array matching predictions length."""
    if isinstance(references, Sequence) and len(references) == len(predictions):
        return references
    return [references] * len(predictions)


@register_reduction("pass@k")
def pass_at_k(references, predictions: Sequence, k: int = 1) -> float:
    """Estimate pass@k from Chen et al. 2021 (https://arxiv.org/abs/2107.03374).

    Args:
        references: The raw target for the document, or a parallel sequence of
            per-repeat references.  A scalar is broadcast to match *predictions*.
        predictions: Per-repeat metric scores for this document.
        k: k in pass@k.
    """
    references = _expand_references(references, predictions)
    n = len(predictions)
    c = len([1 for x, y in zip(predictions, references, strict=True) if x == y])
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def take_first(references, predictions: Sequence):
    """Return the first repeat's prediction, ignoring all others."""
    return predictions[0]
