"""Utilities for document-level BOUQuET translation tasks.

This module exposes:
- a custom dataset loader for YAML-configured ConfigurableTasks
- shared prompt rendering helpers
- paragraph/document reconstruction from sentence-level BOUQuET rows

Supported behavior:
- direct bilingual pair if present
- reverse bilingual pair if only the reverse exists
- otherwise, for non-English↔non-English pairs, reconstruct through
  English-aligned paragraph views

This makes it suitable for:
- English-centric static tasks
- a generic runtime task, e.g. src_lang=hin_Deva, tgt_lang=spa_Latn
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, DefaultDict, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset

EN_CODE = "eng_Latn"

# Friendly names for prompt rendering.
# Unknown codes fall back to the raw code.
LANG_METADATA = {
    "eng_Latn": "English",
    "arb_Arab": "Modern Standard Arabic",
    "arz_Arab": "Egyptian Arabic",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "hin_Deva": "Hindi",
    "ind_Latn": "Indonesian",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "zho_Hans": "Chinese",
    "zho_Hant": "Chinese",
    "cat_Latn": "Catalan",
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "nld_Latn": "Dutch",
    "est_Latn": "Estonian",
    "fin_Latn": "Finnish",
    "ell_Grek": "Greek",
    "hun_Latn": "Hungarian",
    "isl_Latn": "Icelandic",
    "gle_Latn": "Irish",
    "ita_Latn": "Italian",
    "lav_Latn": "Latvian",
    "lit_Latn": "Lithuanian",
    "nob_Latn": "Norwegian Bokmål",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "swe_Latn": "Swedish",
    "ukr_Cyrl": "Ukrainian",
    "bel_Cyrl": "Belarusian",
    "bul_Cyrl": "Bulgarian",
    "srp_Cyrl": "Serbian",
    "hrv_Latn": "Croatian",
    "bos_Latn": "Bosnian",
    "mkd_Cyrl": "Macedonian",
    "mlt_Latn": "Maltese",
    "cym_Latn": "Welsh",
    "eus_Latn": "Basque",
    "glg_Latn": "Galician",
    "ben_Beng": "Bengali",
    "guj_Gujr": "Gujarati",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "nep_Deva": "Nepali",
    "ory_Orya": "Odia",
    "pan_Guru": "Punjabi",
    "sin_Sinh": "Sinhala",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "urd_Arab": "Urdu",
    "asm_Beng": "Assamese",
    "awa_Deva": "Awadhi",
    "bho_Deva": "Bhojpuri",
    "mag_Deva": "Magahi",
    "mai_Deva": "Maithili",
    "san_Deva": "Sanskrit",
    "sat_Beng": "Santali",
    "lus_Latn": "Mizo",
    "mni_Beng": "Manipuri",
    "khm_Khmr": "Khmer",
    "lao_Laoo": "Lao",
    "mya_Mymr": "Burmese",
    "tha_Thai": "Thai",
    "vie_Latn": "Vietnamese",
    "jpn_Jpan": "Japanese",
    "kor_Hang": "Korean",
    "mon_Cyrl": "Mongolian",
    "tur_Latn": "Turkish",
    "kaz_Cyrl": "Kazakh",
    "kir_Cyrl": "Kyrgyz",
    "uzn_Latn": "Uzbek",
    "tat_Cyrl": "Tatar",
    "aze_Latn": "Azerbaijani",
    "heb_Hebr": "Hebrew",
    "pes_Arab": "Persian",
    "tgl_Latn": "Tagalog",
    "hat_Latn": "Haitian Creole",
    "swh_Latn": "Swahili",
    "amh_Ethi": "Amharic",
    "som_Latn": "Somali",
    "yor_Latn": "Yoruba",
    "ibo_Latn": "Igbo",
    "hau_Latn": "Hausa",
    "zul_Latn": "Zulu",
    "xho_Latn": "Xhosa",
    "afr_Latn": "Afrikaans",
}

PROMPT_TEMPLATE = (
    "Translate the following text from {src_lang} into {tgt_lang}.\n"
    "Please write only its translation to {tgt_lang}, without any additional comments.\n"
    "Make sure that your response is a translation to {tgt_lang} and not the original text.\n\n"
    "{src_lang}: {input_text}\n"
    "{tgt_lang}:\n\n"
)


def _id_sort_key(x: str) -> Tuple[Any, ...]:
    """Sort IDs like P2, P10, P458 by numeric component when present."""
    m = re.search(r"\d+", x or "")
    if not m:
        return (x or "", 0)
    return (x[: m.start()], int(m.group()), x[m.end() :])


def _sent_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Best-effort sentence ordering key within a paragraph."""
    if "sent_id" in row and row["sent_id"] not in (None, "", "<na>"):
        return ("sent_id", _id_sort_key(str(row["sent_id"])))

    uid = str(row.get("uniq_id", ""))
    m = re.search(r"(\d+)\s*$", uid)
    if m:
        return ("uniq_tailnum", int(m.group()))

    return ("fallback", 0)


def _lang_name(code: str) -> str:
    return LANG_METADATA.get(code, code)


def render_prompt(*, src_code: str, tgt_code: str, source_text: str) -> str:
    """Render the translation prompt for a supported source/target direction."""
    return PROMPT_TEMPLATE.format(
        src_lang=_lang_name(src_code),
        tgt_lang=_lang_name(tgt_code),
        tgt_code=tgt_code,
        input_text=source_text,
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Render the prompt for a document-level example."""
    src_code = doc.get("src_lang")
    tgt_code = doc.get("tgt_lang")
    if not src_code or not tgt_code:
        raise KeyError("Expected 'src_lang' and 'tgt_lang' in BOUQuET example.")

    return render_prompt(
        src_code=src_code,
        tgt_code=tgt_code,
        source_text=doc.get("source", ""),
    )


def doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the document-level reference translation."""
    return doc["target"]


def _normalize_text(text: str | None) -> str:
    return (text or "").replace("\n", " ").strip()


def _collect_sentence_paragraphs(
    rows: Iterable[Dict[str, Any]],
    *,
    src_code: str,
    tgt_code: str,
    side: str,
) -> Dict[str, List[str]]:
    """Return {paragraph_id: [sentence1, sentence2, ...]} for one side of a pair."""
    if side not in ("src", "tgt"):
        raise ValueError(f"Unsupported side: {side}")

    buckets: DefaultDict[str, List[Tuple[Tuple[Any, ...], str]]] = defaultdict(list)

    for row in rows:
        if row.get("level") != "sentence_level":
            continue
        if row.get("src_lang") != src_code:
            continue
        if row.get("tgt_lang") != tgt_code:
            continue

        text = row.get("src_text") if side == "src" else row.get("tgt_text")
        text = _normalize_text(text)
        if not text:
            continue

        par_id = row.get("par_id") or row.get("uniq_id")
        if not par_id:
            continue

        buckets[str(par_id)].append((_sent_sort_key(row), text))

    out: Dict[str, List[str]] = {}
    for par_id in sorted(buckets.keys(), key=_id_sort_key):
        sents = buckets[par_id]
        sents.sort(key=lambda x: x[0])
        out[par_id] = [text for _, text in sents]

    return out


def _group_pair_to_documents_from_rows(
    rows: Dataset,
    *,
    row_src_lang: str,
    row_tgt_lang: str,
    output_src_lang: str,
    output_tgt_lang: str,
    swap: bool,
) -> Dataset:
    """Reconstruct document-level examples from sentence-level rows.

    Parameters
    ----------
    rows:
        BOUQuET split rows.
    row_src_lang, row_tgt_lang:
        The actual direction stored in the dataset rows.
    output_src_lang, output_tgt_lang:
        The direction exposed to the task.
    swap:
        If True, reverse source/target when reconstructing examples.
    """
    if not swap:
        src_paras = _collect_sentence_paragraphs(
            rows, src_code=row_src_lang, tgt_code=row_tgt_lang, side="src"
        )
        tgt_paras = _collect_sentence_paragraphs(
            rows, src_code=row_src_lang, tgt_code=row_tgt_lang, side="tgt"
        )
    else:
        src_paras = _collect_sentence_paragraphs(
            rows, src_code=row_src_lang, tgt_code=row_tgt_lang, side="tgt"
        )
        tgt_paras = _collect_sentence_paragraphs(
            rows, src_code=row_src_lang, tgt_code=row_tgt_lang, side="src"
        )

    common_par_ids = sorted(set(src_paras) & set(tgt_paras), key=_id_sort_key)

    docs: List[Dict[str, Any]] = []
    for par_id in common_par_ids:
        source_sents = src_paras[par_id]
        target_sents = tgt_paras[par_id]

        docs.append(
            {
                "document_id": par_id,
                "lp": f"{output_src_lang}-{output_tgt_lang}",
                "src_lang": output_src_lang,
                "tgt_lang": output_tgt_lang,
                "source": "\n".join(source_sents),
                "target": "\n".join(target_sents),
                "num_segments": len(source_sents),
            }
        )

    return Dataset.from_list(docs)


def _collect_lang_against_english(
    rows: Dataset,
    *,
    lang: str,
) -> Dict[str, List[str]]:
    """Return {par_id: [sentences]} for one non-English language aligned via English.

    Accepts either underlying storage direction:
    - eng_Latn -> lang  => use tgt side
    - lang -> eng_Latn  => use src side
    """
    forward = _collect_sentence_paragraphs(
        rows,
        src_code=EN_CODE,
        tgt_code=lang,
        side="tgt",
    )

    reverse = _collect_sentence_paragraphs(
        rows,
        src_code=lang,
        tgt_code=EN_CODE,
        side="src",
    )

    if forward and reverse:
        # Prefer forward if both are present; consistency matters more than policy.
        return forward
    if forward:
        return forward
    if reverse:
        return reverse

    raise ValueError(
        f"Language {lang} was not found aligned with English in this split."
    )


def _load_non_english_pair_via_english(
    rows: Dataset,
    *,
    src_lang: str,
    tgt_lang: str,
) -> Dataset:
    """Reconstruct a non-English pair using English-aligned paragraph views."""
    src_paras = _collect_lang_against_english(rows, lang=src_lang)
    tgt_paras = _collect_lang_against_english(rows, lang=tgt_lang)

    common_par_ids = sorted(set(src_paras) & set(tgt_paras), key=_id_sort_key)

    docs: List[Dict[str, Any]] = []
    for par_id in common_par_ids:
        source_sents = src_paras[par_id]
        target_sents = tgt_paras[par_id]

        docs.append(
            {
                "document_id": par_id,
                "lp": f"{src_lang}-{tgt_lang}",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "source": "\n".join(source_sents),
                "target": "\n".join(target_sents),
                "num_segments": len(source_sents),
            }
        )

    return Dataset.from_list(docs)


def _load_split(split: str, **kwargs: Any) -> Dataset:
    """Load one BOUQuET split.

    We intentionally do not forward arbitrary lm-eval task/model kwargs into
    `load_dataset`, because those may contain unrelated keys such as
    `pretrained`, which would break dataset loading.

    If `HF_TOKEN` is present in the environment, it is passed explicitly.
    """
    _ = kwargs

    token = os.environ.get("HF_TOKEN")
    if token:
        return load_dataset("facebook/bouquet", split=split, token=token)
    return load_dataset("facebook/bouquet", split=split)


def _pair_exists(rows: Iterable[Dict[str, Any]], *, src_lang: str, tgt_lang: str) -> bool:
    """Check whether a given pair exists in a split."""
    for row in rows:
        if row.get("level") != "sentence_level":
            continue
        if row.get("src_lang") == src_lang and row.get("tgt_lang") == tgt_lang:
            return True
    return False


def load_bouquet_dataset(
    *,
    src_lang: str,
    tgt_lang: str,
    split: str = "test",
    **kwargs: Any,
) -> Dict[str, Dataset]:
    """Load document-level BOUQuET for an arbitrary direction.

    Resolution strategy:
    1. Use direct bilingual rows if present.
    2. Else use reversed bilingual rows if present.
    3. Else, if both languages are non-English, reconstruct the pair by
       aligning each language against English and intersecting on `par_id`.

    Parameters
    ----------
    src_lang:
        Source language code, e.g. "hin_Deva".
    tgt_lang:
        Target language code, e.g. "spa_Latn".
    split:
        BOUQuET split to load, usually `dev` or `test`.
    **kwargs:
        Extra keyword args accepted for lm-eval compatibility. They are ignored
        when calling `load_dataset`.
    """
    _ = kwargs

    if not src_lang or not tgt_lang:
        raise ValueError("Both src_lang and tgt_lang must be provided.")
    if src_lang == tgt_lang:
        raise ValueError("Source and target languages must differ.")

    rows = _load_split(split)

    # Case 1: direct pair exists
    if _pair_exists(rows, src_lang=src_lang, tgt_lang=tgt_lang):
        ds = _group_pair_to_documents_from_rows(
            rows,
            row_src_lang=src_lang,
            row_tgt_lang=tgt_lang,
            output_src_lang=src_lang,
            output_tgt_lang=tgt_lang,
            swap=False,
        )
        return {split: ds}

    # Case 2: reverse direct pair exists
    if _pair_exists(rows, src_lang=tgt_lang, tgt_lang=src_lang):
        ds = _group_pair_to_documents_from_rows(
            rows,
            row_src_lang=tgt_lang,
            row_tgt_lang=src_lang,
            output_src_lang=src_lang,
            output_tgt_lang=tgt_lang,
            swap=True,
        )
        return {split: ds}

    # Case 3: non-English pair reconstructed via English alignment
    if src_lang != EN_CODE and tgt_lang != EN_CODE:
        ds = _load_non_english_pair_via_english(
            rows,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        return {split: ds}

    raise ValueError(
        f"Could not construct pair {src_lang}-{tgt_lang} from BOUQuET split '{split}'."
    )