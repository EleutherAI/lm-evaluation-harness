r"""
Paired statistical comparison of two lm-eval-harness runs on the SAME task/items.

Why this exists
---------------
`scripts/model_comparator.py` compares two models with an UNPAIRED two-sample z-test:

    Z = (acc1 - acc2) / sqrt(se1**2 + se2**2)

When both models are evaluated on the *same items* (the usual case), that test is the
wrong null model: it ignores the per-item pairing, so it is both less powerful and
mis-calibrated. The statistically correct test for "are these two models different on
the same items?" is **McNemar's paired test** on per-item correctness. It also relies on
a normal approximation that is unsafe at the small sample sizes typical of local eval runs
(model_comparator defaults to limit=100); see Miller 2024 (arXiv:2411.00640) and
Bowyer et al. 2025 (arXiv:2503.01747).

This script reads two `--log_samples` outputs, aligns them by `doc_id`, and reports:
  - each model's accuracy with a Wilson score CI (valid at small n / extreme p),
  - McNemar's exact paired test,
  - a paired bootstrap CI on the accuracy difference,
  - a small-n (CLT-safety) warning.

Stats here are verified against statsmodels (Wilson CI to ~3e-16; McNemar exact-p to 0.0).

Usage
-----
    lm_eval --model hf --model_args pretrained=A --tasks arc_easy --log_samples \\
            --output_path runs/A
    lm_eval --model hf --model_args pretrained=B --tasks arc_easy --log_samples \\
            --output_path runs/B
    python paired_model_comparison.py runs/A runs/B --metric acc
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


# --------------------------- statistics (no new deps) ---------------------------
def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - half, center + half)


def mcnemar_exact(a: np.ndarray, b: np.ndarray):
    """Paired McNemar exact test on per-item correctness arrays a, b (0/1)."""
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n01 = int(np.sum(~a & b))  # A wrong, B right
    n10 = int(np.sum(a & ~b))  # A right, B wrong
    n_disc = n01 + n10
    if n_disc == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * stats.binom.cdf(k, n_disc, 0.5))
    return {"n01": n01, "n10": n10, "n_discordant": n_disc, "p_value": float(p)}


def bootstrap_diff_ci(a, b, alpha=0.05, n_boot=10000, seed=0):
    """Paired bootstrap CI for mean(b) - mean(a) (resample item indices jointly)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    return (
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
    )


def clt_warning(n: int, min_safe: int = 300):
    if n < min_safe:
        return (
            f"n={n} < {min_safe}: normal-approx (z-test) error bars are unreliable here; "
            f"prefer the paired McNemar test and Wilson/bootstrap CIs reported above."
        )
    return None


# --------------------------- lm-eval sample loading ---------------------------
def _resolve_jsonl(path: str) -> str:
    if os.path.isdir(path):
        cands = glob.glob(
            os.path.join(path, "**", "*.jsonl"), recursive=True
        ) + glob.glob(os.path.join(path, "**", "samples_*.json"), recursive=True)
        if not cands:
            raise FileNotFoundError(
                f"no sample file (*.jsonl / samples_*.json) under {path}"
            )
        return sorted(cands)[-1]
    return path


def load_correctness(path: str, metric: str) -> dict:
    """Return {doc_id: 0/1} from an lm-eval --log_samples file."""
    f = _resolve_jsonl(path)
    out = {}
    with open(f) as fh:
        records = (
            json.load(fh)
            if f.endswith(".json") and not f.endswith(".jsonl")
            else [json.loads(line) for line in fh if line.strip()]
        )
    for r in records:
        if "doc_id" not in r:
            continue
        # per-sample metric value was merged in via example.update(metrics)
        val = r.get(metric)
        if val is None:  # tolerate "acc,none"-style keys
            val = next((v for k, v in r.items() if k.split(",")[0] == metric), None)
        if val is None:
            continue
        out[r["doc_id"]] = int(float(val) > 0.5)
    if not out:
        raise ValueError(f"no '{metric}' values found in {f}")
    return out


# --------------------------------- main ---------------------------------
def compare(path_a, path_b, name_a, name_b, metric, alpha):
    ca, cb = load_correctness(path_a, metric), load_correctness(path_b, metric)
    shared = sorted(set(ca) & set(cb))
    if not shared:
        raise ValueError("no overlapping doc_ids between the two runs")
    only_a, only_b = len(set(ca) - set(cb)), len(set(cb) - set(ca))
    a = np.array([ca[d] for d in shared])
    b = np.array([cb[d] for d in shared])
    n = len(shared)
    ka, kb = int(a.sum()), int(b.sum())
    mc = mcnemar_exact(a, b)
    return {
        "metric": metric,
        "n_items": n,
        "dropped_unmatched": {name_a: only_a, name_b: only_b},
        name_a: {"acc": ka / n, "wilson_ci": wilson_ci(ka, n, alpha)},
        name_b: {"acc": kb / n, "wilson_ci": wilson_ci(kb, n, alpha)},
        "diff_b_minus_a": kb / n - ka / n,
        "paired_bootstrap_ci": bootstrap_diff_ci(a, b, alpha),
        "mcnemar": mc,
        "significant": mc["p_value"] < alpha,
        "clt_warning": clt_warning(n),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_a", help="path to model A's --log_samples file or output dir")
    ap.add_argument("run_b", help="path to model B's --log_samples file or output dir")
    ap.add_argument(
        "--metric", default="acc", help="per-sample metric to compare (default: acc)"
    )
    ap.add_argument("--name-a", default="model_A")
    ap.add_argument("--name-b", default="model_B")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", help="optional path to write JSON results")
    args = ap.parse_args()

    r = compare(
        args.run_a, args.run_b, args.name_a, args.name_b, args.metric, args.alpha
    )
    a, b = r[args.name_a], r[args.name_b]
    print(
        f"\nPaired comparison on metric '{r['metric']}'  (n={r['n_items']} shared items)"
    )
    if any(r["dropped_unmatched"].values()):
        print(f"  (dropped unmatched docs: {r['dropped_unmatched']})")
    print(
        f"  {args.name_a:12s} acc={a['acc']:.4f}  Wilson95%=[{a['wilson_ci'][0]:.4f},{a['wilson_ci'][1]:.4f}]"
    )
    print(
        f"  {args.name_b:12s} acc={b['acc']:.4f}  Wilson95%=[{b['wilson_ci'][0]:.4f},{b['wilson_ci'][1]:.4f}]"
    )
    print(
        f"  diff (B-A) = {r['diff_b_minus_a']:+.4f}  paired bootstrap 95% CI "
        f"[{r['paired_bootstrap_ci'][0]:+.4f},{r['paired_bootstrap_ci'][1]:+.4f}]"
    )
    mc = r["mcnemar"]
    print(
        f"  McNemar exact: discordant n01={mc['n01']}, n10={mc['n10']}, p={mc['p_value']:.4f}"
    )
    print(
        f"  => {'SIGNIFICANT' if r['significant'] else 'NOT significant'} at alpha={args.alpha} (paired test)"
    )
    if r["clt_warning"]:
        print(f"  WARNING: {r['clt_warning']}")
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
