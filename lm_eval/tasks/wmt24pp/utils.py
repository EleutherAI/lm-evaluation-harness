"""Utilities for the WMT24++ translation tasks.

This module provides helpers used by YAML-configured ConfigurableTasks. It
exposes the `custom_dataset` loader, along with logic to render the official
WMT24++ prompt template so that all language-pair YAMLs can share a single
`doc_to_text` implementation.
"""

from __future__ import annotations

from typing import Any, Dict

from datasets import Dataset, load_dataset

SRC_LANG = "English"

TARGET_METADATA = {
    "ar_EG": {"tgt_lang": "Arabic", "tgt_region": "Egypt"},
    "ar_SA": {"tgt_lang": "Arabic", "tgt_region": "Saudi Arabia"},
    "bg_BG": {"tgt_lang": "Bulgarian", "tgt_region": "Bulgaria"},
    "bn_IN": {"tgt_lang": "Bengali", "tgt_region": "India"},
    "ca_ES": {"tgt_lang": "Catalan", "tgt_region": "Spain"},
    "cs_CZ": {"tgt_lang": "Czech", "tgt_region": "Czechia"},
    "da_DK": {"tgt_lang": "Danish", "tgt_region": "Denmark"},
    "de_DE": {"tgt_lang": "German", "tgt_region": "Germany"},
    "el_GR": {"tgt_lang": "Greek", "tgt_region": "Greece"},
    "es_MX": {"tgt_lang": "Spanish", "tgt_region": "Mexico"},
    "et_EE": {"tgt_lang": "Estonian", "tgt_region": "Estonia"},
    "fa_IR": {"tgt_lang": "Persian", "tgt_region": "Iran"},
    "fi_FI": {"tgt_lang": "Finnish", "tgt_region": "Finland"},
    "fil_PH": {"tgt_lang": "Filipino", "tgt_region": "Philippines"},
    "fr_CA": {"tgt_lang": "French", "tgt_region": "Canada"},
    "fr_FR": {"tgt_lang": "French", "tgt_region": "France"},
    "gu_IN": {"tgt_lang": "Gujarati", "tgt_region": "India"},
    "he_IL": {"tgt_lang": "Hebrew", "tgt_region": "Israel"},
    "hi_IN": {"tgt_lang": "Hindi", "tgt_region": "India"},
    "hr_HR": {"tgt_lang": "Croatian", "tgt_region": "Croatia"},
    "hu_HU": {"tgt_lang": "Hungarian", "tgt_region": "Hungary"},
    "id_ID": {"tgt_lang": "Indonesian", "tgt_region": "Indonesia"},
    "is_IS": {"tgt_lang": "Icelandic", "tgt_region": "Iceland"},
    "it_IT": {"tgt_lang": "Italian", "tgt_region": "Italy"},
    "ja_JP": {"tgt_lang": "Japanese", "tgt_region": "Japan"},
    "kn_IN": {"tgt_lang": "Kannada", "tgt_region": "India"},
    "ko_KR": {"tgt_lang": "Korean", "tgt_region": "South Korea"},
    "lt_LT": {"tgt_lang": "Lithuanian", "tgt_region": "Lithuania"},
    "lv_LV": {"tgt_lang": "Latvian", "tgt_region": "Latvia"},
    "ml_IN": {"tgt_lang": "Malayalam", "tgt_region": "India"},
    "mr_IN": {"tgt_lang": "Marathi", "tgt_region": "India"},
    "nl_NL": {"tgt_lang": "Dutch", "tgt_region": "Netherlands"},
    "no_NO": {"tgt_lang": "Norwegian", "tgt_region": "Norway"},
    "pa_IN": {"tgt_lang": "Punjabi", "tgt_region": "India"},
    "pl_PL": {"tgt_lang": "Polish", "tgt_region": "Poland"},
    "pt_BR": {"tgt_lang": "Portuguese", "tgt_region": "Brazil"},
    "pt_PT": {"tgt_lang": "Portuguese", "tgt_region": "Portugal"},
    "ro_RO": {"tgt_lang": "Romanian", "tgt_region": "Romania"},
    "ru_RU": {"tgt_lang": "Russian", "tgt_region": "Russia"},
    "sk_SK": {"tgt_lang": "Slovak", "tgt_region": "Slovakia"},
    "sl_SI": {"tgt_lang": "Slovenian", "tgt_region": "Slovenia"},
    "sr_RS": {"tgt_lang": "Serbian", "tgt_region": "Serbia"},
    "sv_SE": {"tgt_lang": "Swedish", "tgt_region": "Sweden"},
    "sw_KE": {"tgt_lang": "Swahili", "tgt_region": "Kenya"},
    "sw_TZ": {"tgt_lang": "Swahili", "tgt_region": "Tanzania"},
    "ta_IN": {"tgt_lang": "Tamil", "tgt_region": "India"},
    "te_IN": {"tgt_lang": "Telugu", "tgt_region": "India"},
    "th_TH": {"tgt_lang": "Thai", "tgt_region": "Thailand"},
    "tr_TR": {"tgt_lang": "Turkish", "tgt_region": "Turkey"},
    "uk_UA": {"tgt_lang": "Ukrainian", "tgt_region": "Ukraine"},
    "ur_PK": {"tgt_lang": "Urdu", "tgt_region": "Pakistan"},
    "vi_VN": {"tgt_lang": "Vietnamese", "tgt_region": "Vietnam"},
    "zh_CN": {"tgt_lang": "Chinese", "tgt_region": "China"},
    "zh_TW": {"tgt_lang": "Chinese", "tgt_region": "Taiwan"},
    "zu_ZA": {"tgt_lang": "Zulu", "tgt_region": "South Africa"},
}

PROMPT_TEMPLATE = (
    "You are a professional {src_lang} to {tgt_lang} translator, tasked with providing "
    "translations suitable for use in {tgt_region} ({tgt_code}). Your goal is to accurately "
    "convey the meaning and nuances of the original {src_lang} text while adhering to {tgt_lang} "
    "grammar, vocabulary, and cultural sensitivities.\n"
    "Please translate the following {src_lang} text into {tgt_lang} ({tgt_code}):\n\n"
    "{input_text}\n\n"
    "Produce only the {tgt_lang} translation, without any additional explanations or commentary:\n\n"
)


def render_prompt(*, lang_pair: str, source_text: str) -> str:
    """Render the official WMT24++ translation prompt for a given language pair."""
    if "-" not in lang_pair:
        msg = f"lang_pair must be of the form 'en-XX_YY', got {lang_pair}"
        raise ValueError(msg)

    _, tgt_code = lang_pair.split("-", maxsplit=1)
    info = TARGET_METADATA.get(tgt_code)
    if info is None:
        msg = (
            f"Unknown WMT24++ target code '{tgt_code}'. Please add metadata to"
            " TARGET_METADATA to render the prompt."
        )
        raise KeyError(msg)

    return PROMPT_TEMPLATE.format(
        src_lang=SRC_LANG,
        tgt_lang=info["tgt_lang"],
        tgt_region=info["tgt_region"],
        tgt_code=tgt_code,
        input_text=source_text,
    )


def doc_to_text(doc: Dict[str, Any]) -> str:
    """Shared doc_to_text function that renders the WMT24++ prompt."""
    lang_pair = doc.get("lp")
    if not lang_pair:
        raise KeyError("Expected 'lp' field in WMT24++ example.")

    source = doc.get("source", "")
    return render_prompt(lang_pair=lang_pair, source_text=source)


def load_wmt24pp_dataset(*, lang_pair: str, split: str = "train", **kwargs: Any) -> Dict[str, Dataset]:
    """Load and filter the WMT24++ dataset for a specific language pair.

    Parameters
    ----------
    lang_pair:
        Exact value of the `lp` field / HF config name, e.g. "en-de_DE".
    split:
        Dataset split name to load. WMT24++ exposes a single split ("train"),
        which we treat as the evaluation split.
    **kwargs:
        Extra keyword arguments forwarded to `load_dataset`. Currently unused
        but accepted for compatibility with ConfigurableTask metadata plumbing.

    Returns
    -------
    dict[str, Dataset]
        Mapping from the requested split name to the filtered dataset.
    """
    _ = kwargs  # ignore extraneous metadata

    ds = load_dataset("google/wmt24pp", lang_pair, split=split)
    ds = ds.filter(lambda ex: not ex["is_bad_source"])
    return {split: ds}
