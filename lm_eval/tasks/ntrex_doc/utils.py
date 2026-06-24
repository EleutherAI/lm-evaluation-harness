"""Utilities for document-level NTREX translation tasks.

This module exposes:
- a custom dataset loader for YAML-configured ConfigurableTasks
- shared prompt rendering helpers
- document reconstruction from NTREX sentence-level text files and DOCUMENT_IDS.tsv

Supported behavior:
- official English -> X tasks
- derived X -> English tasks by swapping source/reference files

If NTREX is not available locally, it is cloned automatically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset

NTREX_GIT_URL = "https://github.com/MicrosoftTranslator/NTREX.git"
NTREX_GIT_REF = os.environ.get("NTREX_GIT_REF", "main")

LANG_METADATA = {
    "eng": "English",
    "deu": "German",
    "fra": "French",
    "spa": "Spanish",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "ces": "Czech",
    "pol": "Polish",
    "ron": "Romanian",
    "hun": "Hungarian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "hrv": "Croatian",
    "srp": "Serbian",
    "bul": "Bulgarian",
    "ukr": "Ukrainian",
    "rus": "Russian",
    "ell": "Greek",
    "tur": "Turkish",
    "ara": "Arabic",
    "heb": "Hebrew",
    "fas": "Persian",
    "hin": "Hindi",
    "ben": "Bengali",
    "urd": "Urdu",
    "tam": "Tamil",
    "tel": "Telugu",
    "mar": "Marathi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "guj": "Gujarati",
    "pan": "Punjabi",
    "ori": "Odia",
    "asm": "Assamese",
    "nep": "Nepali",
    "sin": "Sinhala",
    "tha": "Thai",
    "vie": "Vietnamese",
    "ind": "Indonesian",
    "msa": "Malay",
    "tgl": "Tagalog",
    "jpn": "Japanese",
    "kor": "Korean",
    "zho": "Chinese",
    "mya": "Burmese",
    "khm": "Khmer",
    "lao": "Lao",
    "amh": "Amharic",
    "swh": "Swahili",
    "som": "Somali",
    "hau": "Hausa",
    "yor": "Yoruba",
    "ibo": "Igbo",
    "zul": "Zulu",
    "xho": "Xhosa",
    "afr": "Afrikaans",
}

PROMPT_TEMPLATE = (
    "Translate the following text from {src_lang} into {tgt_lang}.\n"
    "Please write only its translation to {tgt_lang}, without any additional comments.\n"
    "Make sure that your response is a translation to {tgt_lang} and not the original text.\n\n"
    "{src_lang}: {input_text}\n"
    "{tgt_lang}:\n\n"
)


def _lang_name(code: str) -> str:
    return LANG_METADATA.get(code, code)


def render_prompt(*, src_code: str, tgt_code: str, source_text: str) -> str:
    return PROMPT_TEMPLATE.format(
        src_lang=_lang_name(src_code),
        tgt_lang=_lang_name(tgt_code),
        input_text=source_text,
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    src_code = doc.get("src_lang")
    tgt_code = doc.get("tgt_lang")
    if not src_code or not tgt_code:
        raise KeyError("Expected 'src_lang' and 'tgt_lang' in NTREX example.")

    return render_prompt(
        src_code=src_code,
        tgt_code=tgt_code,
        source_text=doc.get("source", ""),
    )


def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["target"]


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
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to clone NTREX into {dest}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{e.stdout}\n"
            f"stderr:\n{e.stderr}"
        ) from e


def _find_ntrex_root() -> Path:
    """Resolve local NTREX checkout root, cloning it if needed."""
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


def _read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NTREX file: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def _group_by_docids(
    src_lines: List[str],
    tgt_lines: List[str],
    docids: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    if not (len(src_lines) == len(tgt_lines) == len(docids)):
        raise ValueError(
            "Mismatched line counts:\n"
            f"  src:    {len(src_lines)}\n"
            f"  tgt:    {len(tgt_lines)}\n"
            f"  docids: {len(docids)}"
        )

    src_docs: Dict[str, List[str]] = defaultdict(list)
    tgt_docs: Dict[str, List[str]] = defaultdict(list)

    doc_order: List[str] = []
    seen = set()

    for src, tgt, docid in zip(src_lines, tgt_lines, docids):
        docid = docid.strip()
        src = src.strip()
        tgt = tgt.strip()

        if not docid:
            continue

        if docid not in seen:
            seen.add(docid)
            doc_order.append(docid)

        src_docs[docid].append(src)
        tgt_docs[docid].append(tgt)

    return src_docs, tgt_docs, doc_order


def _find_ref_file(root: Path, split: str, tgt_lang: str) -> Path:
    candidates = [
        root / f"NTREX-128/{split}-ref.{tgt_lang}.txt",
        root / f"NTREX-additional/{split}-ref.{tgt_lang}.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find NTREX reference file for target language '{tgt_lang}' "
        f"in NTREX-128 or NTREX-additional."
    )


def _load_eng_to_x_documents(*, root: Path, split: str, tgt_lang: str) -> Dataset:
    src_file = root / f"NTREX-128/{split}-src.eng.txt"
    tgt_file = _find_ref_file(root, split, tgt_lang)
    docids_file = root / "DOCUMENT_IDS.tsv"

    src_lines = _read_lines(src_file)
    tgt_lines = _read_lines(tgt_file)
    docids = _read_lines(docids_file)

    src_docs, tgt_docs, doc_order = _group_by_docids(src_lines, tgt_lines, docids)

    docs = []
    for docid in doc_order:
        docs.append(
            {
                "document_id": docid,
                "lp": f"eng-{tgt_lang}",
                "src_lang": "eng",
                "tgt_lang": tgt_lang,
                "source": "\n".join(src_docs[docid]),
                "target": "\n".join(tgt_docs[docid]),
                "num_segments": len(src_docs[docid]),
            }
        )

    return Dataset.from_list(docs)


def _load_x_to_eng_documents(*, root: Path, split: str, src_lang: str) -> Dataset:
    src_file = root / f"NTREX-128/{split}-src.eng.txt"
    tgt_file = _find_ref_file(root, split, src_lang)
    docids_file = root / "DOCUMENT_IDS.tsv"

    eng_lines = _read_lines(src_file)
    x_lines = _read_lines(tgt_file)
    docids = _read_lines(docids_file)

    x_docs, eng_docs, doc_order = _group_by_docids(x_lines, eng_lines, docids)

    docs = []
    for docid in doc_order:
        docs.append(
            {
                "document_id": docid,
                "lp": f"{src_lang}-eng",
                "src_lang": src_lang,
                "tgt_lang": "eng",
                "source": "\n".join(x_docs[docid]),
                "target": "\n".join(eng_docs[docid]),
                "num_segments": len(x_docs[docid]),
            }
        )

    return Dataset.from_list(docs)


def load_ntrex_dataset(
    *,
    src_lang: str,
    tgt_lang: str,
    split: str = "test",
    **kwargs: Any,
) -> Dict[str, Dataset]:
    _ = kwargs

    root = _find_ntrex_root()
    ntrex_split = "newstest2019"

    if src_lang == tgt_lang:
        raise ValueError("Source and target languages must differ.")

    if src_lang == "eng" and tgt_lang != "eng":
        ds = _load_eng_to_x_documents(root=root, split=ntrex_split, tgt_lang=tgt_lang)
        return {split: ds}

    if tgt_lang == "eng" and src_lang != "eng":
        ds = _load_x_to_eng_documents(root=root, split=ntrex_split, src_lang=src_lang)
        return {split: ds}

    raise NotImplementedError(
        "NTREX task family is English-centric only: supported directions are eng->X and X->eng."
    )