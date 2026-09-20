#!/usr/bin/env python3
"""Generate document-level WMT GeneralMT task YAMLs.

This script ensures the data is downloaded and processed before task discovery.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from utils import _default_data_root, _ensure_processed

COMMON_YAML = """# Common configuration for document-level WMT GeneralMT testset translation tasks

training_split: null
validation_split: null
test_split: test

output_type: generate_until

doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target
custom_dataset: !function utils.load_wmt_generalmt_dataset

generation_kwargs:
  temperature: 0.0
  do_sample: false
  max_gen_toks: 2048
  until: []

metric_list:
  - metric: bleu
    aggregation: bleu
    higher_is_better: true
  - metric: chrf
    aggregation: chrf
    higher_is_better: true
"""

PAIR_TEMPLATE = """include: ../wmt_generalmt_doc_common.yaml

task: {task_name}

tag:
  - translation
  - wmt_generalmt_doc

metadata:
  version: 1.0
  testset: "{testset}"
  src_lang: "{src_lang}"
  tgt_lang: "{tgt_lang}"
"""

GROUP_TEMPLATE = """group: {group_name}
task:
{task_entries}
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_year(testset: str) -> str | None:
    m = re.search(r"(20\d{2}|19\d{2})$", testset)
    return m.group(1) if m else None


def _keep_testset(testset: str) -> bool:
    lower = testset.lower()
    if lower.startswith("florestest"):
        return False
    if lower.startswith("wmttest2024"):
        return False
    return lower.startswith("newstest") or lower.startswith("wmttest")


def _discover_specs(root: Path) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []

    for src_file in sorted(root.glob("*.SRC")):
        base = src_file.name[:-4]
        tgt_file = root / f"{base}.TGT"
        if not tgt_file.exists():
            continue

        if "." not in base:
            continue

        testset, pair = base.split(".", maxsplit=1)
        if not _keep_testset(testset):
            continue

        if "-" not in pair:
            continue

        src_lang, tgt_lang = pair.split("-", maxsplit=1)
        specs.append((testset, src_lang, tgt_lang))

    return specs


def main() -> None:
    task_root = Path(__file__).resolve().parent
    tasks_dir = task_root / "tasks"
    groups_dir = task_root / "groups"

    data_root = _default_data_root()
    processed_root = _ensure_processed(data_root)
    specs = _discover_specs(processed_root)

    if not specs:
        raise RuntimeError(
            f"No processed WMT files were discovered under {processed_root}. "
            "Expected files like newstest2010.en-de.SRC and .TGT"
        )

    _write(task_root / "wmt_generalmt_doc_common.yaml", COMMON_YAML)

    all_tasks: list[str] = []
    en_to_x_tasks: list[str] = []
    x_to_en_tasks: list[str] = []
    by_year: dict[str, list[str]] = defaultdict(list)
    by_pair: dict[str, list[str]] = defaultdict(list)

    for testset, src_lang, tgt_lang in specs:
        task_name = f"wmt-generalmt-doc-{testset}-{src_lang}-{tgt_lang}"
        pair_name = f"{src_lang}-{tgt_lang}"
        year = _extract_year(testset)

        all_tasks.append(task_name)
        by_pair[pair_name].append(task_name)

        if src_lang == "en" and tgt_lang != "en":
            en_to_x_tasks.append(task_name)
        if tgt_lang == "en" and src_lang != "en":
            x_to_en_tasks.append(task_name)

        if year is not None:
            by_year[year].append(task_name)

        yaml_text = PAIR_TEMPLATE.format(
            task_name=task_name,
            testset=testset,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        _write(tasks_dir / f"{task_name.replace('-', '_')}.yaml", yaml_text)

    _write(
        groups_dir / "wmt_generalmt_doc_all.yaml",
        GROUP_TEMPLATE.format(
            group_name="wmt-generalmt-doc-all",
            task_entries="\n".join(f"  - {task}" for task in all_tasks),
        ),
    )

    _write(
        groups_dir / "wmt_generalmt_doc_en_all.yaml",
        GROUP_TEMPLATE.format(
            group_name="wmt-generalmt-doc-en-all",
            task_entries="\n".join(f"  - {task}" for task in en_to_x_tasks),
        ),
    )

    _write(
        groups_dir / "wmt_generalmt_doc_all_en.yaml",
        GROUP_TEMPLATE.format(
            group_name="wmt-generalmt-doc-all-en",
            task_entries="\n".join(f"  - {task}" for task in x_to_en_tasks),
        ),
    )

    for year, tasks in sorted(by_year.items()):
        _write(
            groups_dir / f"wmt_generalmt_doc_{year}.yaml",
            GROUP_TEMPLATE.format(
                group_name=f"wmt-generalmt-doc-{year}",
                task_entries="\n".join(f"  - {task}" for task in tasks),
            ),
        )

    for pair_name, tasks in sorted(by_pair.items()):
        safe_pair = pair_name.replace("-", "_")
        _write(
            groups_dir / f"wmt_generalmt_doc_{safe_pair}.yaml",
            GROUP_TEMPLATE.format(
                group_name=f"wmt-generalmt-doc-{pair_name}",
                task_entries="\n".join(f"  - {task}" for task in tasks),
            ),
        )

    print(f"Wrote {len(all_tasks)} pair+testset tasks")
    print(f"Wrote {len(by_pair)} language-pair groups")
    print(f"Wrote {len(by_year)} year groups")
    print("Wrote global groups:")
    print("  - wmt-generalmt-doc-all")
    print("  - wmt-generalmt-doc-en-all")
    print("  - wmt-generalmt-doc-all-en")


if __name__ == "__main__":
    main()