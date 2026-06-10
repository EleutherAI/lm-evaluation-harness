#!/usr/bin/env python3
"""Generate English-centric document-level BOUQuET task YAMLs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set, Tuple

from datasets import load_dataset

EN_CODE = "eng_Latn"

COMMON_YAML = """# Common configuration for document-level BOUQuET translation tasks

dataset_path: facebook/bouquet

training_split: null
validation_split: dev
test_split: test

output_type: generate_until

doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target
custom_dataset: !function utils.load_bouquet_dataset

generation_kwargs:
  temperature: 0.0
  do_sample: false
  max_gen_toks: 2048
  until: []

metric_list:
  - metric: bleu
    aggregation: bleu
    higher_is_better: true
  - metric: ter
    aggregation: ter
    higher_is_better: false
  - metric: chrf
    aggregation: chrf
    higher_is_better: true
"""

PAIR_TEMPLATE = """include: ../bouquet_doc_common.yaml

task: {task_name}

tag:
  - translation
  - bouquet_doc

metadata:
  version: 1.0
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


def _discover_pairs(rows: Iterable[dict]) -> Set[Tuple[str, str]]:
    pairs: Set[Tuple[str, str]] = set()
    for row in rows:
        if row.get("level") != "sentence_level":
            continue
        src = row.get("src_lang")
        tgt = row.get("tgt_lang")
        if src and tgt:
            pairs.add((str(src), str(tgt)))
    return pairs


def main() -> None:
    task_root = Path(__file__).resolve().parent
    en_to_x_dir = task_root / "en_to_x"
    x_to_en_dir = task_root / "x_to_en"
    groups_dir = task_root / "groups"

    _write(task_root / "bouquet_doc_common.yaml", COMMON_YAML)

    ds = load_dataset("facebook/bouquet")
    all_pairs: Set[Tuple[str, str]] = set()
    for split in ("dev", "test"):
        all_pairs.update(_discover_pairs(ds[split]))

    # Any language that appears paired with English on either side
    partner_langs = sorted(
        {
            tgt for src, tgt in all_pairs if src == EN_CODE and tgt != EN_CODE
        }
        | {
            src for src, tgt in all_pairs if tgt == EN_CODE and src != EN_CODE
        }
    )

    en_to_x_tasks: list[str] = []
    x_to_en_tasks: list[str] = []
    bidirectional_tasks: list[str] = []

    for lang in partner_langs:
        forward_task = f"bouquet-doc-en-{lang}"
        reverse_task = f"bouquet-doc-{lang}-en"

        en_to_x_tasks.append(forward_task)
        x_to_en_tasks.append(reverse_task)
        bidirectional_tasks.extend([forward_task, reverse_task])

        forward_yaml = PAIR_TEMPLATE.format(
            task_name=forward_task,
            src_lang=EN_CODE,
            tgt_lang=lang,
        )
        reverse_yaml = PAIR_TEMPLATE.format(
            task_name=reverse_task,
            src_lang=lang,
            tgt_lang=EN_CODE,
        )

        _write(en_to_x_dir / f"{forward_task.replace('-', '_')}.yaml", forward_yaml)
        _write(x_to_en_dir / f"{reverse_task.replace('-', '_')}.yaml", reverse_yaml)

    en_all_yaml = GROUP_TEMPLATE.format(
        group_name="bouquet-doc-en-all",
        task_entries="\n".join(f"  - {task}" for task in en_to_x_tasks),
    )
    all_en_yaml = GROUP_TEMPLATE.format(
        group_name="bouquet-doc-all-en",
        task_entries="\n".join(f"  - {task}" for task in x_to_en_tasks),
    )
    bidirectional_all_yaml = GROUP_TEMPLATE.format(
        group_name="bouquet-doc-bidirectional-all",
        task_entries="\n".join(f"  - {task}" for task in bidirectional_tasks),
    )

    _write(groups_dir / "bouquet_doc_en_all.yaml", en_all_yaml)
    _write(groups_dir / "bouquet_doc_all_en.yaml", all_en_yaml)
    _write(groups_dir / "bouquet_doc_bidirectional_all.yaml", bidirectional_all_yaml)

    print(f"Wrote {len(en_to_x_tasks)} English->X tasks")
    print(f"Wrote {len(x_to_en_tasks)} X->English tasks")
    print("Wrote group YAMLs:")
    print("  - bouquet-doc-en-all")
    print("  - bouquet-doc-all-en")
    print("  - bouquet-doc-bidirectional-all")


if __name__ == "__main__":
    main()