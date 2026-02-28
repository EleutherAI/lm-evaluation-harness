from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from lm_eval.api.registry import register_reduction as reduce


if TYPE_CHECKING:
    from collections.abc import Sequence


def _default_k_values(n: int) -> list[int]:
    """Generate sensible default k values for pass@k given n total samples.

    Dense at small k (where the curve is steepest), log-spaced for larger k.
    Always includes 1 and n.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1]

    # Always-useful small values
    anchors = [1, 3, 5, 10, 20, 50, 100]

    # Log-spaced values between 100 and n
    if n > 100:
        import math

        log_min, log_max = math.log10(100), math.log10(n)
        num_points = max(2, int(log_max - log_min) * 3)  # ~3 points per decade
        anchors += [
            int(10 ** (log_min + i * (log_max - log_min) / num_points))
            for i in range(1, num_points)
        ]

    # Always include n itself
    anchors.append(n)

    # Filter to valid range
    return sorted({k for k in anchors if 1 <= k <= n})


@reduce("pass@k")
def pass_at_k(
    references: Any,
    predictions: Sequence[int] | Sequence[bool],
    *,
    k: list[int] | int | None = None,
) -> dict[str, float]:
    """Estimate pass@k from Chen et al. 2021 (https://arxiv.org/abs/2107.03374).

    Predictions are per-repeat metric scores (e.g. 0/1 from exact_match).
    A truthy score counts as a pass.

    Args:
        references: Unused.
        predictions: Per-repeat metric scores for this document.
        k: k in pass@k. If None, uses sensible defaults based on n.
    """
    k_values = (
        [k]
        if isinstance(k, int)
        else _default_k_values(len(predictions))
        if k is None
        else k
    )

    n = len(predictions)
    c = sum(1 for p in predictions if p)

    results = {}
    for ki in k_values:
        if n - c < ki:
            results[f"pass@{ki}"] = 1.0
        else:
            results[f"pass@{ki}"] = float(
                1.0 - np.prod(1.0 - ki / np.arange(n - c + 1, n + 1))  # type: ignore
            )
    return results
