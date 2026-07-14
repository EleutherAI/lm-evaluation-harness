"""Utilities for document-level WMT GeneralMT testset translation tasks.

This task family automatically:
- downloads the WMT GeneralMT zip if missing
- extracts XML files
- converts them into document-level .SRC / .TGT plain-text files

Expected processed output layout:
<data_root>/processed/<testset>.<src>-<tgt>.SRC
<data_root>/processed/<testset>.<src>-<tgt>.TGT

Documents are serialized as:
- one sentence per line
- blank lines separating documents
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple
from zipfile import ZipFile

from datasets import Dataset

WMT_URL = "https://data.statmt.org/wmt24/general-mt/wmt24_GeneralMT-devsets.zip"
WMT_ZIP_NAME = "wmt24_GeneralMT-devsets.zip"

LANG_METADATA = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "cs": "Czech",
    "pl": "Polish",
    "ro": "Romanian",
    "hu": "Hungarian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "hr": "Croatian",
    "sr": "Serbian",
    "bg": "Bulgarian",
    "uk": "Ukrainian",
    "ru": "Russian",
    "el": "Greek",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "fa": "Persian",
    "hi": "Hindi",
    "bn": "Bengali",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "si": "Sinhala",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
    "tl": "Tagalog",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "my": "Burmese",
    "km": "Khmer",
    "lo": "Lao",
    "am": "Amharic",
    "sw": "Swahili",
    "so": "Somali",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "zu": "Zulu",
    "xh": "Xhosa",
    "af": "Afrikaans",
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
        raise KeyError("Expected 'src_lang' and 'tgt_lang' in WMT example.")

    return render_prompt(
        src_code=src_code,
        tgt_code=tgt_code,
        source_text=doc.get("source", ""),
    )


def doc_to_target(doc: Dict[str, Any]) -> str:
    return doc["target"]


def _default_data_root() -> Path:
    env_path = os.environ.get("WMT_GENERALMT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    lm_eval_data_dir = os.environ.get("LM_EVAL_DATA_DIR")
    if lm_eval_data_dir:
        return Path(lm_eval_data_dir).expanduser().resolve() / "wmt_generalmt"

    return (Path.cwd() / "data" / "raw" / "wmt_generalmt").resolve()


def _download_zip(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(WMT_URL) as response, zip_path.open("wb") as out:
        shutil.copyfileobj(response, out)


def _extract_zip(zip_path: Path, xml_root: Path) -> None:
    xml_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(xml_root)


def _normalize_segment_text(text: str) -> str:
    return " ".join((text or "").split())


def _parse_xml_to_documents(xml_path: Path) -> Tuple[List[List[str]], List[List[str]]]:
    """Parse WMT XML into source and target document sentence lists.

    Supported layouts include:
    1) <dataset><doc>...</doc></dataset>
    2) <dataset><collection><doc>...</doc></collection></dataset>

    Each <doc> is expected to contain:
    - one <src ...> block
    - one or more <ref ...> blocks

    We use the first <ref> block.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    docs = root.findall("./doc")
    if not docs:
        docs = root.findall("./collection/doc")
    if not docs:
        docs = root.findall(".//doc")

    if not docs:
        raise ValueError(f"Could not find any <doc> elements in {xml_path}")

    src_docs: List[List[str]] = []
    tgt_docs: List[List[str]] = []

    for doc in docs:
        src_node = doc.find("./src")
        ref_nodes = doc.findall("./ref")
        ref_node = ref_nodes[0] if ref_nodes else None

        if src_node is None or ref_node is None:
            continue

        src_sents = [
            _normalize_segment_text("".join(seg.itertext()))
            for seg in src_node.findall(".//seg")
        ]
        tgt_sents = [
            _normalize_segment_text("".join(seg.itertext()))
            for seg in ref_node.findall(".//seg")
        ]

        if not src_sents or not tgt_sents:
            continue

        src_docs.append(src_sents)
        tgt_docs.append(tgt_sents)

    if not src_docs:
        raise ValueError(f"No aligned documents were parsed from {xml_path}")

    if len(src_docs) != len(tgt_docs):
        raise ValueError(f"Mismatched parsed doc counts in {xml_path}")

    return src_docs, tgt_docs


def _write_docs(docs: List[List[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for i, sents in enumerate(docs):
            for sent in sents:
                out.write(sent.rstrip() + "\n")
            if i != len(docs) - 1:
                out.write("\n")


def _ensure_processed(root: Path) -> Path:
    """Ensure the dataset is downloaded, extracted, and converted."""
    xml_root = root / "xml"
    processed_root = root / "processed"
    zip_path = root / WMT_ZIP_NAME
    sentinel = processed_root / ".complete"

    if sentinel.exists():
        return processed_root

    if not zip_path.exists():
        _download_zip(zip_path)

    if not xml_root.exists() or not any(xml_root.rglob("*.xml")):
        _extract_zip(zip_path, xml_root)

    processed_root.mkdir(parents=True, exist_ok=True)

    for xml_path in sorted(xml_root.rglob("*.xml")):
        base = xml_path.stem  # e.g. newstest2010.en-de
        if "." not in base:
            continue
        testset, pair = base.split(".", maxsplit=1)
        if "-" not in pair:
            continue
        src_lang, tgt_lang = pair.split("-", maxsplit=1)

        src_docs, tgt_docs = _parse_xml_to_documents(xml_path)
        _write_docs(src_docs, processed_root / f"{testset}.{src_lang}-{tgt_lang}.SRC")
        _write_docs(tgt_docs, processed_root / f"{testset}.{src_lang}-{tgt_lang}.TGT")

    sentinel.write_text("ok\n", encoding="utf-8")
    return processed_root


def _read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed WMT file: {path}")
    return path.read_text(encoding="utf-8")


def _split_docs(text: str) -> List[List[str]]:
    docs: List[List[str]] = []
    current: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                docs.append(current)
                current = []
            continue
        current.append(line)

    if current:
        docs.append(current)

    return docs


def _load_processed_pair(
    *,
    root: Path,
    testset: str,
    src_lang: str,
    tgt_lang: str,
) -> Dataset:
    stem = f"{testset}.{src_lang}-{tgt_lang}"
    src_file = root / f"{stem}.SRC"
    tgt_file = root / f"{stem}.TGT"

    src_docs = _split_docs(_read_text(src_file))
    tgt_docs = _split_docs(_read_text(tgt_file))

    if len(src_docs) != len(tgt_docs):
        raise ValueError(
            f"Mismatched document counts for {stem}: "
            f"{len(src_docs)} source docs vs {len(tgt_docs)} target docs"
        )

    docs = []
    for idx, (src_sents, tgt_sents) in enumerate(zip(src_docs, tgt_docs)):
        docs.append(
            {
                "document_id": f"{stem}.doc{idx}",
                "testset": testset,
                "lp": f"{src_lang}-{tgt_lang}",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": "\n".join(src_sents),
                "target": "\n".join(tgt_sents),
                "num_segments": len(src_sents),
            }
        )

    return Dataset.from_list(docs)


def load_wmt_generalmt_dataset(
    *,
    src_lang: str,
    tgt_lang: str,
    testset: str,
    split: str = "test",
    **kwargs: Any,
) -> Dict[str, Dataset]:
    _ = kwargs

    if not src_lang or not tgt_lang:
        raise ValueError("Both src_lang and tgt_lang must be provided.")
    if src_lang == tgt_lang:
        raise ValueError("Source and target languages must differ.")
    if not testset:
        raise ValueError("testset must be provided.")

    root = _default_data_root()
    processed_root = _ensure_processed(root)

    ds = _load_processed_pair(
        root=processed_root,
        testset=testset,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
    return {split: ds}