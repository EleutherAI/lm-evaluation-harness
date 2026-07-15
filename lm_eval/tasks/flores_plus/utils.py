"""
Utilities for FLORES+ translation tasks.

FLORES+ stores each language variety in a separate config with a ``text`` field.
Parallel sentences share the same ``id`` within a split. This module joins two
language configs into the ``sentence_{lang}`` fields used by task YAMLs.
"""

from __future__ import annotations

import json
import logging
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


eval_logger = logging.getLogger(__name__)

DATASET_REPO = "openlanguagedata/flores_plus"
ENG_LANG = "eng_Latn"
EVAL_SPLITS = ("dev", "devtest")


@lru_cache(maxsize=1)
def _split_languages() -> dict[str, frozenset[str]]:
    """Map each FLORES+ split to the language configs available on Hugging Face."""
    from huggingface_hub import list_repo_files

    langs: dict[str, set[str]] = {split: set() for split in EVAL_SPLITS}
    for path in list_repo_files(DATASET_REPO, repo_type="dataset"):
        for split in EVAL_SPLITS:
            prefix = f"{split}/"
            if path.startswith(prefix) and path.endswith(".jsonl"):
                langs[split].add(path.split("/", 1)[1].removesuffix(".jsonl"))
    return {split: frozenset(values) for split, values in langs.items()}


def _languages_for_split(split: str) -> frozenset[str]:
    try:
        return _split_languages()[split]
    except KeyError as exc:
        raise ValueError(f"Unknown FLORES+ split '{split}'.") from exc


def _languages_for_pair(src_lang: str, tgt_lang: str) -> list[str]:
    return [
        split
        for split in EVAL_SPLITS
        if src_lang in _languages_for_split(split)
        and tgt_lang in _languages_for_split(split)
    ]


@lru_cache(maxsize=1)
def _language_names() -> dict[str, str]:
    """Build config code -> display name mapping from the dataset README."""
    import re

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET_REPO, "README.md", repo_type="dataset")
    text = Path(path).read_text(encoding="utf-8")
    rows: list[tuple[str, str, str, str]] = []

    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5 or parts[1] in ("Code", "`Code`"):
            continue
        iso = parts[1].strip("`")
        script = parts[2].strip("`")
        glotto = parts[3].strip("`").split()[0]
        name = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", parts[4])
        if len(iso) != 3 or len(script) != 4 or not glotto:
            continue
        rows.append((iso, script, glotto, name))

    configs = set(list_language_configs())

    names: dict[str, str] = {}
    for config in configs:
        parts = config.split("_")
        iso, script = parts[0], parts[1]
        glotto = parts[2] if len(parts) > 2 else None
        if glotto:
            for iso_row, script_row, glotto_row, name in rows:
                if iso == iso_row and script == script_row and glotto == glotto_row:
                    names[config] = name
                    break
            continue

        matches = [
            name
            for iso_row, script_row, _, name in rows
            if iso_row == iso and script_row == script
        ]
        if matches:
            names[config] = matches[0]

    return names


def language_display_name(code: str) -> str:
    """Return a human-readable language name for a FLORES+ config code."""
    names = _language_names()
    if code in names:
        return names[code]

    iso639 = code.split("_")[0]
    try:
        import pycountry

        lang = pycountry.languages.get(alpha_3=iso639)
        if lang is not None:
            return lang.name
    except ImportError:
        pass
    return code


@cache
def _load_lang_texts(lang: str, split: str) -> dict[int, str]:
    path = hf_hub_download(
        DATASET_REPO,
        f"{split}/{lang}.jsonl",
        repo_type="dataset",
    )
    texts: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts[int(row["id"])] = row["text"]
    return texts


def _join_languages(src_lang: str, tgt_lang: str, split: str) -> list[dict[str, Any]]:
    src_texts = _load_lang_texts(src_lang, split)
    tgt_texts = _load_lang_texts(tgt_lang, split)
    common_ids = sorted(set(src_texts) & set(tgt_texts))
    if not common_ids:
        raise ValueError(
            f"No aligned sentences for {src_lang}->{tgt_lang} in split '{split}'"
        )
    src_key = f"sentence_{src_lang}"
    tgt_key = f"sentence_{tgt_lang}"
    return [
        {"id": doc_id, src_key: src_texts[doc_id], tgt_key: tgt_texts[doc_id]}
        for doc_id in common_ids
    ]


def load_dataset(**kwargs):
    """
    Load aligned FLORES+ translation pairs for ``src_lang`` and ``tgt_lang``.

    ``src_lang`` and ``tgt_lang`` are passed via per-task ``dataset_kwargs``.
    """
    import datasets

    src_lang = kwargs.pop("src_lang", None)
    tgt_lang = kwargs.pop("tgt_lang", None)
    kwargs.pop("version", None)

    if not src_lang or not tgt_lang:
        raise ValueError(
            "FLORES+ tasks require `src_lang` and `tgt_lang` in dataset_kwargs."
        )

    eval_logger.info("Loading FLORES+ pair %s -> %s", src_lang, tgt_lang)
    splits = _languages_for_pair(src_lang, tgt_lang)
    if "devtest" not in splits:
        raise ValueError(
            f"FLORES+ pair {src_lang}->{tgt_lang} is missing devtest data for one "
            "or both languages. Regenerate tasks with generate_tasks.py to exclude "
            "languages that are not present in both dev and devtest."
        )
    return {
        split: datasets.Dataset.from_list(_join_languages(src_lang, tgt_lang, split))
        for split in splits
    }


def list_language_configs() -> list[str]:
    """List FLORES+ language configs present in both dev and devtest."""
    splits = _split_languages()
    return sorted(splits["dev"] & splits["devtest"])
