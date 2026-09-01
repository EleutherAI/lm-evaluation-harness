"""Utilities for WMT24++ document-level translation tasks.

This module supports:
- official English -> X tasks from WMT24++
- derived X -> English tasks by swapping source/target from the aligned
  English -> X configs

It exposes:
- `custom_dataset` loader for YAML-configured ConfigurableTasks
- shared `doc_to_text` and `doc_to_target` helpers
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from datasets import Dataset, load_dataset

LANG_METADATA = {
    "en": {"lang": "English", "region": None},
    "ar_EG": {"lang": "Arabic", "region": "Egypt"},
    "ar_SA": {"lang": "Arabic", "region": "Saudi Arabia"},
    "bg_BG": {"lang": "Bulgarian", "region": "Bulgaria"},
    "bn_IN": {"lang": "Bengali", "region": "India"},
    "ca_ES": {"lang": "Catalan", "region": "Spain"},
    "cs_CZ": {"lang": "Czech", "region": "Czechia"},
    "da_DK": {"lang": "Danish", "region": "Denmark"},
    "de_DE": {"lang": "German", "region": "Germany"},
    "el_GR": {"lang": "Greek", "region": "Greece"},
    "es_MX": {"lang": "Spanish", "region": "Mexico"},
    "et_EE": {"lang": "Estonian", "region": "Estonia"},
    "fa_IR": {"lang": "Persian", "region": "Iran"},
    "fi_FI": {"lang": "Finnish", "region": "Finland"},
    "fil_PH": {"lang": "Filipino", "region": "Philippines"},
    "fr_CA": {"lang": "French", "region": "Canada"},
    "fr_FR": {"lang": "French", "region": "France"},
    "gu_IN": {"lang": "Gujarati", "region": "India"},
    "he_IL": {"lang": "Hebrew", "region": "Israel"},
    "hi_IN": {"lang": "Hindi", "region": "India"},
    "hr_HR": {"lang": "Croatian", "region": "Croatia"},
    "hu_HU": {"lang": "Hungarian", "region": "Hungary"},
    "id_ID": {"lang": "Indonesian", "region": "Indonesia"},
    "is_IS": {"lang": "Icelandic", "region": "Iceland"},
    "it_IT": {"lang": "Italian", "region": "Italy"},
    "ja_JP": {"lang": "Japanese", "region": "Japan"},
    "kn_IN": {"lang": "Kannada", "region": "India"},
    "ko_KR": {"lang": "Korean", "region": "South Korea"},
    "lt_LT": {"lang": "Lithuanian", "region": "Lithuania"},
    "lv_LV": {"lang": "Latvian", "region": "Latvia"},
    "ml_IN": {"lang": "Malayalam", "region": "India"},
    "mr_IN": {"lang": "Marathi", "region": "India"},
    "nl_NL": {"lang": "Dutch", "region": "Netherlands"},
    "no_NO": {"lang": "Norwegian", "region": "Norway"},
    "pa_IN": {"lang": "Punjabi", "region": "India"},
    "pl_PL": {"lang": "Polish", "region": "Poland"},
    "pt_BR": {"lang": "Portuguese", "region": "Brazil"},
    "pt_PT": {"lang": "Portuguese", "region": "Portugal"},
    "ro_RO": {"lang": "Romanian", "region": "Romania"},
    "ru_RU": {"lang": "Russian", "region": "Russia"},
    "sk_SK": {"lang": "Slovak", "region": "Slovakia"},
    "sl_SI": {"lang": "Slovenian", "region": "Slovenia"},
    "sr_RS": {"lang": "Serbian", "region": "Serbia"},
    "sv_SE": {"lang": "Swedish", "region": "Sweden"},
    "sw_KE": {"lang": "Swahili", "region": "Kenya"},
    "sw_TZ": {"lang": "Swahili", "region": "Tanzania"},
    "ta_IN": {"lang": "Tamil", "region": "India"},
    "te_IN": {"lang": "Telugu", "region": "India"},
    "th_TH": {"lang": "Thai", "region": "Thailand"},
    "tr_TR": {"lang": "Turkish", "region": "Turkey"},
    "uk_UA": {"lang": "Ukrainian", "region": "Ukraine"},
    "ur_PK": {"lang": "Urdu", "region": "Pakistan"},
    "vi_VN": {"lang": "Vietnamese", "region": "Vietnam"},
    "zh_CN": {"lang": "Chinese", "region": "China"},
    "zh_TW": {"lang": "Chinese", "region": "Taiwan"},
    "zu_ZA": {"lang": "Zulu", "region": "South Africa"},
}

LANGUAGE_PAIRS = (
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ",
    "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR",
    "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR", "en-gu_IN", "en-he_IL",
    "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
    "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR",
    "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK", "en-sl_SI", "en-sr_RS",
    "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW",
    "en-zu_ZA",
)

SUPPORTED_X_LANGS = tuple(lp.split("-", 1)[1] for lp in LANGUAGE_PAIRS)

PROMPT_TEMPLATE = (
    "You are a professional {src_lang} to {tgt_lang} translator, tasked with providing "
    "translations suitable for use in {tgt_region} ({tgt_code}). Your goal is to accurately "
    "convey the meaning and nuances of the original {src_lang} text while adhering to {tgt_lang} "
    "grammar, vocabulary, and cultural sensitivities.\n"
    "Please translate the following {src_lang} text into {tgt_lang} ({tgt_code}):\n\n"
    "{input_text}\n\n"
    "Produce only the {tgt_lang} translation, without any additional explanations or commentary:\n\n"
)


def _lang_info(code: str) -> Dict[str, str | None]:
    info = LANG_METADATA.get(code)
    if info is None:
        raise KeyError(f"Unknown language code: {code}")
    return info


def render_prompt(*, src_code: str, tgt_code: str, source_text: str) -> str:
    """Render the translation prompt for an arbitrary supported source/target direction."""
    src_info = _lang_info(src_code)
    tgt_info = _lang_info(tgt_code)

    tgt_region = tgt_info["region"]
    if tgt_region is None:
        tgt_region = tgt_info["lang"]

    return PROMPT_TEMPLATE.format(
        src_lang=src_info["lang"],
        tgt_lang=tgt_info["lang"],
        tgt_region=tgt_region,
        tgt_code=tgt_code,
        input_text=source_text,
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Render the prompt for a document-level example."""
    src_code = doc.get("src_lang")
    tgt_code = doc.get("tgt_lang")
    if not src_code or not tgt_code:
        raise KeyError("Expected 'src_lang' and 'tgt_lang' in WMT24++ example.")

    source = doc.get("source", "")
    return render_prompt(src_code=src_code, tgt_code=tgt_code, source_text=source)


def doc_to_target(doc: Dict[str, Any]) -> str:
    """Return the document-level reference translation."""
    return doc["target"]


def _group_segment_rows_to_documents(
    rows: Dataset,
    *,
    src_lang: str,
    tgt_lang: str,
    source_field: str,
    target_field: str,
) -> Dataset:
    """Collapse segment-level rows into document-level examples."""
    docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        docs[row["document_id"]].append(row)

    doc_rows = []
    for document_id, segs in docs.items():
        segs = sorted(segs, key=lambda x: x["segment_id"])

        doc_rows.append(
            {
                "document_id": document_id,
                "lp": f"{src_lang}-{tgt_lang}",
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "domain": segs[0].get("domain"),
                "source": "\n".join(seg[source_field].rstrip("\n") for seg in segs),
                "target": "\n".join(seg[target_field].rstrip("\n") for seg in segs),
                "segment_ids": [seg["segment_id"] for seg in segs],
                "num_segments": len(segs),
            }
        )

    doc_rows.sort(key=lambda x: x["document_id"])
    return Dataset.from_list(doc_rows)


def _load_official_en_to_x_dataset(
    *,
    tgt_lang: str,
    split: str,
) -> Dataset:
    """Load the official WMT24++ en->X segment dataset and filter bad sources."""
    hf_config = f"en-{tgt_lang}"
    ds = load_dataset("google/wmt24pp", hf_config, split=split)
    ds = ds.filter(lambda ex: not ex["is_bad_source"])
    return ds


def _load_en_to_x_doc_dataset(
    *,
    tgt_lang: str,
    split: str,
) -> Dataset:
    """Build document-level dataset for en->X."""
    rows = _load_official_en_to_x_dataset(tgt_lang=tgt_lang, split=split)
    return _group_segment_rows_to_documents(
        rows,
        src_lang="en",
        tgt_lang=tgt_lang,
        source_field="source",
        target_field="target",
    )


def _load_x_to_en_doc_dataset(
    *,
    src_lang: str,
    split: str,
) -> Dataset:
    """Build derived document-level dataset for X->en by swapping fields.

    This uses the official en->X config and treats the X-side translation as
    the source text, and the original English source as the reference target.
    """
    rows = _load_official_en_to_x_dataset(tgt_lang=src_lang, split=split)
    return _group_segment_rows_to_documents(
        rows,
        src_lang=src_lang,
        tgt_lang="en",
        source_field="target",
        target_field="source",
    )


def load_wmt24pp_dataset(
    *,
    src_lang: str,
    tgt_lang: str,
    split: str = "train",
    **kwargs: Any,
) -> Dict[str, Dataset]:
    """Load WMT24++ at document level for supported directions.

    Supported directions:
    - en -> X (official)
    - X -> en (derived by swapping the aligned en->X config)

    Parameters
    ----------
    src_lang:
        Source language code, e.g. "en" or "ca_ES".
    tgt_lang:
        Target language code, e.g. "ca_ES" or "en".
    split:
        Dataset split name. WMT24++ exposes only "train", which is treated as
        the evaluation split.
    **kwargs:
        Extra keyword arguments accepted for ConfigurableTask compatibility.

    Returns
    -------
    dict[str, Dataset]
        Mapping from split name to document-level dataset.
    """
    _ = kwargs

    if src_lang == "en" and tgt_lang in SUPPORTED_X_LANGS:
        ds = _load_en_to_x_doc_dataset(tgt_lang=tgt_lang, split=split)
        return {split: ds}

    if tgt_lang == "en" and src_lang in SUPPORTED_X_LANGS:
        ds = _load_x_to_en_doc_dataset(src_lang=src_lang, split=split)
        return {split: ds}

    raise NotImplementedError(
        "Supported directions are only en->X and X->en for WMT24++."
        f" Got src_lang={src_lang!r}, tgt_lang={tgt_lang!r}."
    )
