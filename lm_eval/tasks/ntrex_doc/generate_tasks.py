#!/usr/bin/env python3
"""Generate English-centric document-level NTREX task YAMLs."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

NTREX_GIT_URL = "https://github.com/MicrosoftTranslator/NTREX.git"
NTREX_GIT_REF = os.environ.get("NTREX_GIT_REF", "main")

COMMON_YAML = """# Common configuration for document-level NTREX translation tasks

training_split: null
validation_split: null
test_split: test

output_type: generate_until

doc_to_text: !function utils.doc_to_text
doc_to_target: !function utils.doc_to_target
custom_dataset: !function utils.load_ntrex_dataset

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

PAIR_TEMPLATE = """include: ../ntrex_doc_common.yaml

task: {task_name}

tag:
  - translation
  - ntrex_doc

metadata:
  version: 1.0
  src_lang: "{src_lang}"
  tgt_lang: "{tgt_lang}"
"""

REVERSE_PAIR_TEMPLATE = """include: ../ntrex_doc_common.yaml

task: {task_name}

tag:
  - translation
  - ntrex_doc
  - reverse_derived

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


def _default_ntrex_cache_dir() -> Path:
    lm_eval_data_dir = os.environ.get("LM_EVAL_DATA_DIR")
    if lm_eval_data_dir:
        return Path(lm_eval_data_dir).expanduser().resolve() / "NTREX"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "lm_eval" / "NTREX"

    return Path.home().resolve() / ".cache" / "lm_eval" / "NTREX"


def _ensure_git_available() -> None:
    if shutil.which("git") is None:
        raise RuntimeError(
            "git is required to automatically clone NTREX, but it was not found in PATH."
        )


def _clone_ntrex(dest: Path) -> None:
    _ensure_git_available()
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        NTREX_GIT_REF,
        NTREX_GIT_URL,
        str(dest),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone NTREX into {dest}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e


def _find_ntrex_root() -> Path:
    env_path = os.environ.get("NTREX_PATH")
    if env_path:
        root = Path(env_path).expanduser().resolve()
    else:
        root = _default_ntrex_cache_dir()

    required = [
        root / "DOCUMENT_IDS.tsv",
        root / "NTREX-128",
    ]

    if not all(p.exists() for p in required):
        if root.exists() and any(root.iterdir()):
            missing = [str(p) for p in required if not p.exists()]
            raise FileNotFoundError(
                f"NTREX directory exists at {root}, but required files are missing: {missing}"
            )
        _clone_ntrex(root)

    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required NTREX path after clone: {p}")

    return root


def _discover_target_langs(root: Path) -> list[str]:
    langs = set()

    for base in ("NTREX-128", "NTREX-additional"):
        folder = root / base
        if not folder.exists():
            continue
        for path in folder.glob("newstest2019-ref.*.txt"):
            parts = path.name.split(".")
            # newstest2019-ref.spa.txt -> ["newstest2019-ref", "spa", "txt"]
            if len(parts) >= 3:
                langs.add(parts[-2])

    langs.discard("eng")
    return sorted(langs)


def main() -> None:
    task_root = Path(__file__).resolve().parent
    en_to_x_dir = task_root / "en_to_x"
    x_to_en_dir = task_root / "x_to_en"
    groups_dir = task_root / "groups"

    root = _find_ntrex_root()
    target_langs = _discover_target_langs(root)

    print("NTREX root:", root)
    print(f"Discovered {len(target_langs)} target languages")

    if not target_langs:
        raise RuntimeError(
            f"No target languages were discovered under {root}. "
            "Check that the NTREX repo contains newstest2019-ref.*.txt files."
        )

    _write(task_root / "ntrex_doc_common.yaml", COMMON_YAML)

    en_to_x_tasks: list[str] = []
    x_to_en_tasks: list[str] = []
    bidirectional_tasks: list[str] = []

    for tgt_lang in target_langs:
        forward_task = f"ntrex-doc-eng-{tgt_lang}"
        reverse_task = f"ntrex-doc-{tgt_lang}-eng"

        en_to_x_tasks.append(forward_task)
        x_to_en_tasks.append(reverse_task)
        bidirectional_tasks.extend([forward_task, reverse_task])

        forward_yaml = PAIR_TEMPLATE.format(
            task_name=forward_task,
            src_lang="eng",
            tgt_lang=tgt_lang,
        )
        reverse_yaml = REVERSE_PAIR_TEMPLATE.format(
            task_name=reverse_task,
            src_lang=tgt_lang,
            tgt_lang="eng",
        )

        _write(en_to_x_dir / f"{forward_task.replace('-', '_')}.yaml", forward_yaml)
        _write(x_to_en_dir / f"{reverse_task.replace('-', '_')}.yaml", reverse_yaml)

    en_all_yaml = GROUP_TEMPLATE.format(
        group_name="ntrex-doc-en-all",
        task_entries="\n".join(f"  - {task}" for task in en_to_x_tasks),
    )
    all_en_yaml = GROUP_TEMPLATE.format(
        group_name="ntrex-doc-all-en",
        task_entries="\n".join(f"  - {task}" for task in x_to_en_tasks),
    )
    bidirectional_all_yaml = GROUP_TEMPLATE.format(
        group_name="ntrex-doc-bidirectional-all",
        task_entries="\n".join(f"  - {task}" for task in bidirectional_tasks),
    )

    _write(groups_dir / "ntrex_doc_en_all.yaml", en_all_yaml)
    _write(groups_dir / "ntrex_doc_all_en.yaml", all_en_yaml)
    _write(groups_dir / "ntrex_doc_bidirectional_all.yaml", bidirectional_all_yaml)

    print(f"Wrote {len(en_to_x_tasks)} English->X tasks")
    print(f"Wrote {len(x_to_en_tasks)} X->English tasks")
    print("Wrote group YAMLs:")
    print("  - ntrex-doc-en-all")
    print("  - ntrex-doc-all-en")
    print("  - ntrex-doc-bidirectional-all")


if __name__ == "__main__":
    main()