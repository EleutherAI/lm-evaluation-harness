"""
Generate per-language task YAMLs and per-language group YAMLs for PolyMath.

Produces (18 languages × 4 difficulties) = 72 task YAMLs, e.g.:
    polymath_ar_low.yaml
    polymath_ar_medium.yaml
    ...
    polymath_zh_high.yaml
    polymath_zh_top.yaml

Plus one group YAML per language, e.g.:
    polymath_en.yaml   (groups polymath_en_{low,medium,high,top})

Plus one group YAML per difficulty tier, e.g.:
    polymath_top.yaml   (groups polymath_{ar,bn,de,en,es,...,vi,zh}_{top})

Plus one top-level group YAML:
    polymath.yaml      (groups all 18 per-language groups)

Total: (18 × 4) + 18 + 4 + 1 = 95 YAML configuration files.

Usage:
    python _generate_configs.py
    python _generate_configs.py --output-dir /path/to/output
"""

import argparse
import os

import yaml


# ── Constants ────────────────────────────────────────────────────────────────

LANGUAGES = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "id",
    "it",
    "ja",
    "ko",
    "ms",
    "pt",
    "ru",
    "sw",
    "te",
    "th",
    "vi",
    "zh",
]

DIFFICULTIES = ["low", "medium", "high", "top"]

TEMPLATE_NAME = "_template_yaml"


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_task_yaml(language: str, difficulty: str, split: str) -> dict:
    """Single task config for one language × difficulty combination."""
    return {
        "include": TEMPLATE_NAME,
        "task": f"polymath_{language}_{difficulty}",
        "dataset_name": language,
        "test_split": split,
    }


def make_language_group_yaml(language: str) -> dict:
    """Group YAML that bundles the 4 difficulty tiers for one language."""
    return {
        "group": f"polymath_{language}",
        "task": [f"polymath_{language}_{d}" for d in DIFFICULTIES],
        "aggregate_metric_list": [
            {
                "metric": "math_verify",
                "aggregation": "!function utils.aggregate_dw_acc",
                "weight_by_size": False,
            }
        ],
        "metadata": {"version": 1},
    }


def make_difficulty_group_yaml(difficulty: str) -> dict:
    """Group YAML that bundles all 18 languages for one difficulty tier."""
    return {
        "group": f"polymath_{difficulty}",
        "task": [f"polymath_{language}_{difficulty}" for language in LANGUAGES],
        "aggregate_metric_list": [
            {
                "metric": "math_verify",
                "aggregation": "mean",
                "weight_by_size": False,
            }
        ],
        "metadata": {"version": 1},
    }


def make_top_level_group_yaml() -> dict:
    """Top-level group that bundles all 18 per-language groups."""
    return {
        "group": "polymath",
        "task": [f"polymath_{language}" for language in LANGUAGES],
        "metadata": {"version": 1},
    }


# ── YAML serialisation ────────────────────────────────────────────────────────


class _PolyMathDumper(yaml.Dumper):
    """Custom dumper that avoids mutating the global yaml.Dumper singleton."""

    pass


# Represent None as bare null (matches harness convention)
_PolyMathDumper.add_representer(
    type(None),
    lambda dumper, _: dumper.represent_scalar("tag:yaml.org,2002:null", ""),
)


def _dump(data: dict) -> str:
    """Serialise to YAML, preserving insertion order, no flow style."""
    return yaml.dump(
        data,
        Dumper=_PolyMathDumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def _fix_function_tag(text: str) -> str:
    """
    yaml.dump cannot natively emit custom YAML tags on mapping values.
    Post-process the one known case so lm-eval can parse it:
        aggregation: '!function utils.aggregate_dw_acc'
    →   aggregation: !function utils.aggregate_dw_acc
    """
    return text.replace(
        "aggregation: '!function utils.aggregate_dw_acc'",
        "aggregation: !function utils.aggregate_dw_acc",
    )


# ── Writer ────────────────────────────────────────────────────────────────────


def write_yaml(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        print(f"  overwriting {path}")
    raw = _fix_function_tag(_dump(data))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(raw)
    print(f"  wrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def generate(output_dir: str) -> None:
    total = 0

    # Generate 72 task YAMLs  (language × difficulty)
    for language in LANGUAGES:
        for difficulty in DIFFICULTIES:
            data = make_task_yaml(language, difficulty, difficulty)
            fname = f"polymath_{language}_{difficulty}.yaml"
            write_yaml(os.path.join(output_dir, fname), data)
            total += 1

    # Generate 18 per-language group YAMLs
    for language in LANGUAGES:
        data = make_language_group_yaml(language)
        fname = f"polymath_{language}.yaml"
        write_yaml(os.path.join(output_dir, fname), data)
        total += 1

    # Generate 4 per-difficulty group YAMLs
    for difficulty in DIFFICULTIES:
        data = make_difficulty_group_yaml(difficulty)
        fname = f"polymath_{difficulty}.yaml"
        write_yaml(os.path.join(output_dir, fname), data)
        total += 1

    # Generate 1 top-level group YAML
    data = make_top_level_group_yaml()
    write_yaml(os.path.join(output_dir, "polymath.yaml"), data)
    total += 1

    print(f"\nDone — {total} files written to '{output_dir}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PolyMath lm-eval task YAML configs."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write generated YAML files into (default: current dir).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(args.output_dir)
