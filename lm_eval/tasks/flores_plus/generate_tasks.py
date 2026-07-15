"""
Generate FLORES+ translation task YAMLs.

FLORES+ has 200+ language varieties, so configs are generated programmatically
instead of checked in for every possible pair.

Only English-centric pairs (X <-> English) are checked into git by default.
Use --pairs or --all-pairs for other combinations locally.

Examples:
    # Default: English <-> every other language, both directions
    python generate_tasks.py --overwrite

    # Explicit language pairs (bidirectional by default; not checked in)
    python generate_tasks.py --overwrite --pairs fra_Latn:deu_Latn

    # All ordered language pairs (very large; not checked in)
    python generate_tasks.py --overwrite --all-pairs
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import yaml


_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from utils import ENG_LANG, language_display_name, list_language_configs


TASK_PREFIX = "floresp"
TASK_TAG = "floresp_tasks"
BASE_TEMPLATE = TASK_PREFIX


def task_name(src_lang: str, tgt_lang: str) -> str:
    return f"{TASK_PREFIX}_{src_lang}-{tgt_lang}"


def _parse_pair(spec: str) -> tuple[str, str]:
    """
    Parse a language pair spec.

    Preferred format uses ``:`` as separator (unambiguous for glottocode codes).
    ``-`` is also accepted and splits from the right.
    """
    for sep, split_fn in ((":", str.split), (",", str.split), ("-", str.rsplit)):
        if sep not in spec:
            continue
        left, right = split_fn(spec, sep, 1)
        left, right = left.strip(), right.strip()
        if left and right:
            return left, right
    raise ValueError(
        f"Could not parse pair '{spec}'. Expected 'src:tgt', 'src,tgt', or 'src-tgt'."
    )


def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return list(dict.fromkeys(pairs))


def _alias_lang(lang: str) -> str:
    parts = lang.split("_")
    if len(parts) == 2:
        return f"{parts[0]}_{parts[1]}"
    return lang


def task_alias(src_lang: str, tgt_lang: str) -> str:
    return f"{_alias_lang(src_lang)}->{_alias_lang(tgt_lang)}"


def prompt_text(src_lang: str, tgt_lang: str) -> str:
    src_name = language_display_name(src_lang)
    tgt_name = language_display_name(tgt_lang)
    return (
        f"Translate the following {src_name} text to {tgt_name}.\n"
        f"Output only the translation. Do not explain, list alternatives, "
        f"or add preamble.\n\n"
        f"{src_name}: {{{{sentence_{src_lang}}}}}\n{tgt_name}: "
    )


def language_pairs(
    languages: list[str],
    *,
    all_pairs: bool,
    explicit_pairs: list[tuple[str, str]] | None,
    bidirectional_pairs: bool,
) -> list[tuple[str, str]]:
    if explicit_pairs is not None:
        ordered = explicit_pairs[:]
        if bidirectional_pairs:
            ordered = ordered + [(b, a) for (a, b) in explicit_pairs]
        return _dedupe_pairs([(src, tgt) for src, tgt in ordered if src != tgt])

    if all_pairs:
        return [(src, tgt) for src, tgt in itertools.permutations(languages, 2)]

    non_english = [lang for lang in languages if lang != ENG_LANG]
    return _dedupe_pairs(
        [pair for lang in non_english for pair in ((lang, ENG_LANG), (ENG_LANG, lang))]
    )


def ensure_base_template(output_dir: Path) -> None:
    base_path = output_dir / BASE_TEMPLATE
    if base_path.exists():
        return
    yaml.safe_dump(
        {
            "tag": [TASK_TAG],
            "include": "_flores_plus_common_yaml",
        },
        base_path.open("w", encoding="utf-8"),
        allow_unicode=True,
        sort_keys=False,
    )


def write_group_file(output_dir: Path) -> None:
    group_path = output_dir / "flores_plus.yaml"
    with group_path.open("w", encoding="utf-8") as f:
        f.write("group: flores_plus\n")
        f.write("group_alias: FLORES+\n")
        f.write("task:\n")
        f.write(f"  - tag: {TASK_TAG}\n")
        f.write("aggregate_metric_list:\n")
        f.write("  - metric: bleu\n")
        f.write("    aggregation: mean\n")
        f.write("    weight_by_size: false\n")
        f.write("  - metric: chrf\n")
        f.write("    aggregation: mean\n")
        f.write("    weight_by_size: false\n")
        f.write("  - metric: ter\n")
        f.write("    aggregation: mean\n")
        f.write("    weight_by_size: false\n")
        f.write("metadata:\n")
        f.write("  version: 1.0\n")


def gen_lang_yamls(
    output_dir: Path,
    *,
    overwrite: bool,
    languages: list[str] | None,
    pairs: list[tuple[str, str]] | None,
    bidirectional_pairs: bool,
    all_pairs: bool,
) -> None:
    configs = list_language_configs()
    if ENG_LANG not in configs:
        raise RuntimeError(f"Expected '{ENG_LANG}' in FLORES+ language configs.")

    selected = configs if languages is None else languages
    unknown = sorted(set(selected) - set(configs))
    if unknown:
        raise ValueError(f"Unknown FLORES+ language configs: {', '.join(unknown)}")

    if pairs is not None:
        flat = sorted({lang for pair in pairs for lang in pair})
        unknown_pairs = sorted(set(flat) - set(configs))
        if unknown_pairs:
            raise ValueError(
                "Unknown FLORES+ language configs in --pairs: "
                + ", ".join(unknown_pairs)
            )

    ensure_base_template(output_dir)
    err: list[str] = []

    for src_lang, tgt_lang in language_pairs(
        selected,
        all_pairs=all_pairs,
        explicit_pairs=pairs,
        bidirectional_pairs=bidirectional_pairs,
    ):
        file_name = f"{task_name(src_lang, tgt_lang)}.yaml"
        out_path = output_dir / file_name

        try:
            with out_path.open("w" if overwrite else "x", encoding="utf-8") as f:
                f.write("# Generated by generate_tasks.py\n")
                yaml.safe_dump(
                    {
                        "include": BASE_TEMPLATE,
                        "task": task_name(src_lang, tgt_lang),
                        "task_alias": task_alias(src_lang, tgt_lang),
                        "dataset_kwargs": {
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                        },
                        "doc_to_target": f"sentence_{tgt_lang}",
                        "doc_to_text": prompt_text(src_lang, tgt_lang),
                    },
                    f,
                    allow_unicode=True,
                    sort_keys=False,
                )
        except FileExistsError:
            err.append(file_name)

    write_group_file(output_dir)

    if err:
        raise FileExistsError(
            "Files were not created because they already exist "
            f"(use --overwrite): {', '.join(err)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLORES+ task YAMLs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_TASK_DIR,
        help="Directory to write YAML files to",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Subset of FLORES+ language configs for English-centric generation",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        help=(
            "Explicit language pairs to generate, e.g. "
            "'fra_Latn:deu_Latn' 'zho_Hans:jpn_Jpan'"
        ),
    )
    parser.add_argument(
        "--ordered-pairs",
        action="store_true",
        help="With --pairs, do not generate the reverse direction",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Generate all ordered language pairs instead of only English-centric pairs",
    )
    args = parser.parse_args()

    if args.all_pairs and (args.languages or args.pairs):
        parser.error("--all-pairs cannot be combined with --languages or --pairs")
    if args.pairs and args.languages:
        parser.error("--pairs cannot be combined with --languages")

    parsed_pairs = None if not args.pairs else [_parse_pair(p) for p in args.pairs]

    gen_lang_yamls(
        args.output_dir,
        overwrite=args.overwrite,
        languages=args.languages,
        pairs=parsed_pairs,
        bidirectional_pairs=bool(args.pairs) and (not args.ordered_pairs),
        all_pairs=args.all_pairs,
    )


if __name__ == "__main__":
    main()
