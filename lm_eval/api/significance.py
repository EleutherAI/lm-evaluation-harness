"""Paired significance testing for comparing two evaluation runs.

`lm_eval` reports a per-task standard error for each metric, but it has no
*principled* way to ask whether the gap between two models (or two runs of the
same model) is statistically meaningful. The existing ``scripts/model_comparator.py``
helper answers this with an **unpaired** two-sample z-test::

    Z = (acc_a - acc_b) / sqrt(se_a**2 + se_b**2)

That formula is only valid when the two accuracy estimates are independent. In
`lm_eval` both models are evaluated on *the same documents*, so the per-document
outcomes are paired and usually positively correlated (hard items are hard for
both models). The variance of the difference is

    Var(A - B) = Var(A) + Var(B) - 2 * Cov(A, B),

and dropping the ``-2 * Cov(A, B)`` term inflates the standard error of the
difference, making the unpaired test conservative / underpowered. The textbook
remedy for comparing two classifiers on a shared test set is a *paired* test
(McNemar's test for binary metrics, a paired bootstrap for arbitrary metrics);
see Dietterich (1998), "Approximate Statistical Tests for Comparing Supervised
Classification Learning Algorithms".

This module provides those paired tests as small, dependency-free functions
(``numpy`` + stdlib only -- ``scipy`` is intentionally avoided since it is not a
core dependency). They operate on the per-document scores that `lm_eval` already
writes to ``samples_*.jsonl`` when ``--log_samples`` is set, so a comparison can
be run on saved logs without re-loading any model.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


__all__ = [
    "BootstrapResult",
    "McNemarResult",
    "PairedComparison",
    "compare_paired",
    "mcnemar_test",
    "normal_ci",
    "paired_bootstrap_test",
]


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard-normal CDF (quantile function) via Acklam's rational
    approximation. Dependency-free stand-in for ``scipy.stats.norm.ppf`` with a
    relative error below ~1e-9, which is far finer than any confidence level a
    caller would specify.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in the open interval (0, 1)")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def normal_ci(
    mean: float, stderr: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Two-sided normal-approximation confidence interval ``mean +/- z * stderr``.

    This is the interval implied by the standard errors `lm_eval` already
    reports. It is a Wald interval and can misbehave for accuracies very close to
    0 or 1; for those cases prefer :func:`paired_bootstrap_test`, which makes no
    normality assumption.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in the open interval (0, 1)")
    z = _inv_norm_cdf(0.5 + confidence / 2.0)
    half = z * stderr
    return mean - half, mean + half


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of McNemar's test for a pair of binary (0/1) score vectors.

    ``n_a_only`` counts documents that A got right and B got wrong; ``n_b_only``
    the reverse. Concordant documents (both right or both wrong) carry no
    information about the difference and are ignored by the test.
    """

    n_a_only: int
    n_b_only: int
    statistic: float | None
    p_value: float
    method: str

    @property
    def n_discordant(self) -> int:
        return self.n_a_only + self.n_b_only


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of a paired bootstrap on the per-document score difference."""

    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float
    iters: int


@dataclass(frozen=True)
class PairedComparison:
    """Unified result of comparing two aligned score vectors.

    Always carries the two means, their difference (``a - b``), and a paired
    bootstrap confidence interval on that difference. ``p_value`` comes from the
    test named in ``method`` -- McNemar's test for binary metrics, the paired
    bootstrap otherwise.
    """

    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float
    method: str
    mcnemar: McNemarResult | None = None
    bootstrap: BootstrapResult | None = None

    @property
    def significant(self) -> bool:
        """Whether the difference is significant at ``1 - confidence``."""
        return self.p_value < (1.0 - self.confidence)


def _binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value: under H0 the ``b`` discordant successes
    are ``Binomial(b + c, 0.5)``. Summed in closed form with ``math.comb``.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return min(1.0, 2.0 * tail)


def mcnemar_test(
    n_a_only: int,
    n_b_only: int,
    *,
    exact: bool | None = None,
    continuity_correction: bool = True,
) -> McNemarResult:
    """McNemar's test for two paired binary classifiers.

    Args:
        n_a_only: documents A got right and B got wrong (discordant ``b``).
        n_b_only: documents A got wrong and B got right (discordant ``c``).
        exact: use the exact binomial test. Defaults to ``True`` when there are
            at most 25 discordant pairs (where the chi-square approximation is
            unreliable) and ``False`` otherwise.
        continuity_correction: apply Edwards' continuity correction to the
            chi-square statistic. Ignored by the exact test.
    """
    b, c = int(n_a_only), int(n_b_only)
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "mcnemar_exact")

    if exact is None:
        exact = n <= 25

    if exact:
        return McNemarResult(b, c, None, _binom_two_sided_p(b, c), "mcnemar_exact")

    delta = abs(b - c)
    if continuity_correction:
        delta = max(0.0, delta - 1.0)
    statistic = (delta * delta) / n
    # Survival function of a chi-square with 1 dof: P(X > x) = erfc(sqrt(x / 2)).
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    method = "mcnemar_chi2_cc" if continuity_correction else "mcnemar_chi2"
    return McNemarResult(b, c, statistic, p_value, method)


def _resampled_diff_means(diffs: np.ndarray, iters: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the mean paired difference.

    Resamples document *indices* with replacement (so A and B always share the
    same resampled documents -- this is what makes the bootstrap paired and what
    preserves the cross-model correlation). Chunked to bound peak memory at a few
    million floats regardless of ``n`` or ``iters``.
    """
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]
    out = np.empty(iters, dtype=np.float64)
    chunk = max(1, min(iters, 2_000_000 // max(1, n)))
    done = 0
    while done < iters:
        size = min(chunk, iters - done)
        idx = rng.integers(0, n, size=(size, n))
        out[done : done + size] = diffs[idx].mean(axis=1)
        done += size
    return out


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    iters: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> BootstrapResult:
    """Paired bootstrap test on the difference of two aligned score vectors.

    Works for any per-document metric (binary or continuous). The percentile
    confidence interval is computed on the resampled mean difference; the
    two-sided p-value is the bootstrap achieved significance level for
    ``H0: mean(a - b) == 0``, using the ``(1 + count) / (1 + iters)`` estimator
    (Davison & Hinkley) so it is never exactly zero.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1-D and the same length")
    if a.size == 0:
        raise ValueError("cannot test empty score vectors")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in the open interval (0, 1)")

    diffs = a - b
    observed = float(diffs.mean())
    boot = _resampled_diff_means(diffs, iters, seed)

    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])

    # Center the bootstrap distribution at the null and measure how often a
    # resample is at least as extreme as what we observed.
    centered = boot - observed
    at_least_as_extreme = int(np.count_nonzero(np.abs(centered) >= abs(observed)))
    p_value = (1.0 + at_least_as_extreme) / (1.0 + iters)

    return BootstrapResult(
        diff=observed,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=min(1.0, p_value),
        confidence=confidence,
        iters=iters,
    )


def _is_binary(x: np.ndarray) -> bool:
    return bool(np.all((x == 0.0) | (x == 1.0)))


def compare_paired(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    method: Literal["auto", "mcnemar", "bootstrap"] = "auto",
    iters: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> PairedComparison:
    """Compare two aligned per-document score vectors with a paired test.

    The two sequences must be aligned document-for-document (e.g. ordered by the
    same ``doc_id``). With ``method="auto"`` (the default) McNemar's test is used
    when both vectors are binary and the paired bootstrap otherwise. A paired
    bootstrap confidence interval on the mean difference is always reported, even
    when the p-value comes from McNemar's test.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("scores_a and scores_b must be 1-D and the same length")
    if a.size == 0:
        raise ValueError("cannot compare empty score vectors")

    boot = paired_bootstrap_test(a, b, iters=iters, confidence=confidence, seed=seed)

    use_mcnemar = method == "mcnemar" or (
        method == "auto" and _is_binary(a) and _is_binary(b)
    )
    if method == "mcnemar" and not (_is_binary(a) and _is_binary(b)):
        raise ValueError("McNemar's test requires binary (0/1) scores")

    mcnemar = None
    if use_mcnemar:
        n_a_only = int(np.count_nonzero((a == 1.0) & (b == 0.0)))
        n_b_only = int(np.count_nonzero((a == 0.0) & (b == 1.0)))
        mcnemar = mcnemar_test(n_a_only, n_b_only)
        p_value = mcnemar.p_value
        chosen = mcnemar.method
    else:
        p_value = boot.p_value
        chosen = "paired_bootstrap"

    return PairedComparison(
        n=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        diff=boot.diff,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        p_value=p_value,
        confidence=confidence,
        method=chosen,
        mcnemar=mcnemar,
        bootstrap=boot,
    )
