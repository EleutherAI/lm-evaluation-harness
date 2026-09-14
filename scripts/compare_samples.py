r"""Compare two `lm_eval` runs with a paired significance test.

Unlike ``scripts/model_comparator.py`` -- which re-runs both models and applies
an *unpaired* z-test to the two accuracies -- this script works entirely from the
``samples_*.jsonl`` files that `lm_eval` writes with ``--log_samples``. Because
both runs scored the *same* documents, the per-document outcomes are paired, and
this script uses the correct paired test (McNemar for binary metrics, a paired
bootstrap otherwise) via :mod:`lm_eval.api.significance`. No model is loaded, so
it runs on CPU in well under a second.

Example::

    lm_eval --model hf --model_args pretrained=A --tasks arc_easy \\
        --log_samples --output_path runs/A
    lm_eval --model hf --model_args pretrained=B --tasks arc_easy \\
        --log_samples --output_path runs/B

    python scripts/compare_samples.py runs/A runs/B --metric acc
"""

import argparse
import glob
import json
import os

from lm_eval.api.significance import compare_paired


def _resolve_samples_file(path: str, task: str | None) -> str:
    """Return a samples ``.jsonl`` path, expanding a directory if needed."""
    if os.path.isfile(path):
        return path
    pattern = f"samples_{task}_*.jsonl" if task else "samples_*.jsonl"
    matches = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f"no file matching {pattern!r} found under {path!r}; pass --task or a direct path"
        )
    if len(matches) > 1:
        raise ValueError(
            "multiple samples files found; narrow with --task or pass a direct file path:\n  "
            + "\n  ".join(matches)
        )
    return matches[0]


def load_scores(
    path: str, metric: str, filter_name: str, task: str | None = None
) -> dict[int, float]:
    """Load ``{doc_id: score}`` for one metric/filter from a samples file."""
    samples_path = _resolve_samples_file(path, task)
    scores: dict[int, float] = {}
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("filter", "none") != filter_name:
                continue
            if metric not in row:
                raise KeyError(
                    f"metric {metric!r} not found in {samples_path}; "
                    f"available per-sample metrics: {row.get('metrics')}"
                )
            scores[int(row["doc_id"])] = float(row[metric])
    if not scores:
        raise ValueError(f"no rows with filter={filter_name!r} found in {samples_path}")
    return scores


def align(
    scores_a: dict[int, float], scores_b: dict[int, float]
) -> tuple[list[float], list[float], int]:
    """Align two ``{doc_id: score}`` maps on their shared documents."""
    common = sorted(set(scores_a) & set(scores_b))
    a = [scores_a[i] for i in common]
    b = [scores_b[i] for i in common]
    dropped = (len(scores_a) - len(common)) + (len(scores_b) - len(common))
    return a, b, dropped


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", help="samples_*.jsonl file or output dir for run A")
    parser.add_argument("run_b", help="samples_*.jsonl file or output dir for run B")
    parser.add_argument(
        "--metric", default="acc", help="per-sample metric (default: acc)"
    )
    parser.add_argument("--filter", default="none", help="filter name (default: none)")
    parser.add_argument(
        "--task",
        default=None,
        help="task name, to disambiguate when passing directories",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "mcnemar", "bootstrap"],
        help="paired test to use (default: auto)",
    )
    parser.add_argument(
        "--iters", type=int, default=10_000, help="bootstrap iterations"
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main():
    args = parse_args()
    a_scores = load_scores(args.run_a, args.metric, args.filter, args.task)
    b_scores = load_scores(args.run_b, args.metric, args.filter, args.task)
    a, b, dropped = align(a_scores, b_scores)
    if dropped:
        print(f"warning: {dropped} non-shared document(s) dropped before pairing")

    res = compare_paired(
        a,
        b,
        method=args.method,
        iters=args.iters,
        confidence=args.confidence,
        seed=args.seed,
    )

    pct = round(res.confidence * 100)
    print(f"\nPaired comparison on {res.n} shared documents (metric={args.metric!r})")
    print(f"  run A mean : {res.mean_a:.4f}")
    print(f"  run B mean : {res.mean_b:.4f}")
    print(f"  difference : {res.diff:+.4f}  (A - B)")
    print(
        f"  {pct}% CI     : [{res.ci_low:+.4f}, {res.ci_high:+.4f}]  (paired bootstrap)"
    )
    print(f"  test       : {res.method}")
    print(f"  p-value    : {res.p_value:.4g}")
    verdict = "significant" if res.significant else "not significant"
    print(f"  verdict    : {verdict} at alpha={1 - res.confidence:.2g}")


if __name__ == "__main__":
    main()
