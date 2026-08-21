import logging
import os

import datasets
from huggingface_hub import snapshot_download


eval_logger = logging.getLogger(__name__)

_LANG_CODE_TO_NAME = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt_BR": "Portuguese (Brazil)",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "zh_CN": "Chinese (Simplified)",
    "zh_TW": "Chinese (Traditional)",
}

_META_COLUMNS = {
    "id",
    "movie_id",
    "segment_id",
    "doc_id",
    "line_number",
    "source",
    "timestamp_start",
    "timestamp_end",
    "speaker",
    "translation",
}

# In-process caches to avoid reloading the same dataset split for each subtask.
_DATASET_SPLIT_CACHE = {}
_LANGUAGES_CACHE = {}


def _safe_str(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _select_split(dataset_obj, split, dataset_ref):
    if isinstance(dataset_obj, datasets.DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if len(dataset_obj) == 1:
            only_split = next(iter(dataset_obj.keys()))
            eval_logger.warning(
                "Requested split %s not found in %s. Falling back to only split %s.",
                split,
                dataset_ref,
                only_split,
            )
            return dataset_obj[only_split]
        raise ValueError(
            "Split {} not found in dataset. Available splits: {}".format(
                split, list(dataset_obj.keys())
            )
        )

    if isinstance(dataset_obj, datasets.Dataset):
        eval_logger.warning(
            "Loaded a Dataset (not DatasetDict) from %s. Using it directly.",
            dataset_ref,
        )
        return dataset_obj

    raise TypeError(
        "Unsupported dataset object type from {}: {}".format(dataset_ref, type(dataset_obj))
    )


def _load_split(dataset_ref, split):
    ref = _safe_str(dataset_ref).strip()
    if not ref:
        raise ValueError("Empty dataset reference provided.")
    split_name = _safe_str(split).strip()
    cache_key = (ref, split_name)

    cached = _DATASET_SPLIT_CACHE.get(cache_key)
    if cached is not None:
        eval_logger.debug("Using cached dataset for %s (split=%s).", ref, split_name)
        return cached

    expanded_ref = os.path.expanduser(ref)
    if os.path.exists(expanded_ref):
        dataset_obj = datasets.load_from_disk(expanded_ref)
        selected = _select_split(dataset_obj, split_name, expanded_ref)
        _DATASET_SPLIT_CACHE[cache_key] = selected
        return selected

    if ref.startswith("/") or ref.startswith(".") or ref.startswith("~"):
        raise ValueError("Local dataset path does not exist: {}".format(ref))

    # HF dataset id: first try downloading snapshot and opening as save_to_disk.
    try:
        snapshot_path = snapshot_download(repo_id=ref, repo_type="dataset")
        dataset_obj = datasets.load_from_disk(snapshot_path)
        selected = _select_split(dataset_obj, split_name, ref)
        _DATASET_SPLIT_CACHE[cache_key] = selected
        return selected
    except Exception:
        # Fallback for standard Hub datasets published in load_dataset format.
        try:
            dataset = datasets.load_dataset(ref, split=split_name)
            selected = _select_split(dataset, split_name, ref)
            _DATASET_SPLIT_CACHE[cache_key] = selected
            return selected
        except Exception:
            try:
                dataset_obj = datasets.load_dataset(ref)
                selected = _select_split(dataset_obj, split_name, ref)
                _DATASET_SPLIT_CACHE[cache_key] = selected
                return selected
            except Exception as exc:
                raise ValueError(
                    "Failed to load dataset reference {}. "
                    "Provide a valid local save_to_disk path or a HF dataset id.".format(ref)
                ) from exc


def _infer_languages(dataset):
    if "translation" in dataset.column_names:
        translation_feature = dataset.features.get("translation")
        if hasattr(translation_feature, "keys"):
            return sorted(list(translation_feature.keys()))

        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Cannot infer language keys from translation.")

        first_translation = dataset[0].get("translation", {})
        if not isinstance(first_translation, dict) or not first_translation:
            raise ValueError("Column translation exists but does not contain a non-empty dict.")
        return sorted(list(first_translation.keys()))

    return sorted([col for col in dataset.column_names if col not in _META_COLUMNS])


def _lang_display_name(lang_code):
    code = _safe_str(lang_code)
    if code in _LANG_CODE_TO_NAME:
        return _LANG_CODE_TO_NAME[code]

    normalized = code.replace("-", "_")
    if normalized in _LANG_CODE_TO_NAME:
        return _LANG_CODE_TO_NAME[normalized]

    if "_" in normalized:
        base, variant = normalized.split("_", 1)
        if base in _LANG_CODE_TO_NAME:
            return "{} ({})".format(_LANG_CODE_TO_NAME[base], variant)

    return code


def _extract_pair(doc, src_lang, tgt_lang):
    if "translation" in doc and isinstance(doc["translation"], dict):
        src_text = _safe_str(doc["translation"].get(src_lang, ""))
        tgt_text = _safe_str(doc["translation"].get(tgt_lang, ""))
        return src_text, tgt_text

    src_text = _safe_str(doc.get(src_lang, ""))
    tgt_text = _safe_str(doc.get(tgt_lang, ""))
    return src_text, tgt_text


def load_opensubtitles_parallel(**kwargs):
    """
    Custom dataset loader for OpenSubtitles multi-aligned data.

    Expected kwargs (from metadata/dataset_kwargs):
      - dataset_dir: local save_to_disk path OR HF dataset id
      - split: split name to read from source dataset (default: devtest)
      - output_split: split name returned to harness (default: split)
      - src_lang: source language key (e.g. en)
      - tgt_lang: target language key (e.g. fi)
      - allow_empty: keep examples where src or tgt is empty (default: False)
      - max_samples: optional int to truncate dataset for quick debugging
    """
    dataset_ref = kwargs.get("dataset_dir") or kwargs.get("dataset_repo")
    if not dataset_ref:
        raise ValueError("dataset_dir must be provided in metadata or dataset_kwargs.")

    split = kwargs.get("split", "devtest")
    output_split = kwargs.get("output_split", split)
    src_lang = kwargs.get("src_lang")
    tgt_lang = kwargs.get("tgt_lang")
    allow_empty = bool(kwargs.get("allow_empty", False))
    max_samples = kwargs.get("max_samples")

    if not src_lang or not tgt_lang:
        raise ValueError("Both src_lang and tgt_lang must be provided.")

    dataset = _load_split(dataset_ref, split)
    split_name = _safe_str(split).strip()
    language_cache_key = (_safe_str(dataset_ref).strip(), split_name)
    languages = _LANGUAGES_CACHE.get(language_cache_key)
    if languages is None:
        languages = _infer_languages(dataset)
        _LANGUAGES_CACHE[language_cache_key] = languages

    missing = [lang for lang in (src_lang, tgt_lang) if lang not in languages]
    if missing:
        raise ValueError(
            "Missing language(s) {} in dataset. Available languages: {}".format(
                missing, languages
            )
        )

    eval_logger.info(
        "Loading OpenSubtitles translation direction %s -> %s from %s (split=%s)",
        src_lang,
        tgt_lang,
        dataset_ref,
        split,
    )

    def _has_valid_pair(doc):
        src_text, tgt_text = _extract_pair(doc, src_lang, tgt_lang)
        if allow_empty:
            return True
        return bool(src_text.strip()) and bool(tgt_text.strip())

    dataset = dataset.filter(_has_valid_pair)

    def _map_doc(doc):
        src_text, tgt_text = _extract_pair(doc, src_lang, tgt_lang)

        segment_raw = doc.get("segment_id", doc.get("line_number", -1))
        try:
            segment_id = int(segment_raw)
        except (TypeError, ValueError):
            segment_id = -1

        return {
            "id": _safe_str(doc.get("id", "")),
            "movie_id": _safe_str(doc.get("movie_id", doc.get("doc_id", ""))),
            "segment_id": segment_id,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "src": src_text,
            "tgt": tgt_text,
        }

    dataset = dataset.map(_map_doc, remove_columns=dataset.column_names)

    if max_samples is not None:
        max_samples = int(max_samples)
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    eval_logger.info(
        "Prepared %d aligned samples for %s -> %s.",
        len(dataset),
        src_lang,
        tgt_lang,
    )

    return {output_split: dataset}


def doc_to_text(doc):
    src_lang_name = _lang_display_name(doc.get("src_lang", ""))
    tgt_lang_name = _lang_display_name(doc.get("tgt_lang", ""))
    return (
        "Translate the following sentence from {} to {}:\n{}\nTranslation:\n".format(
            src_lang_name, tgt_lang_name, doc["src"]
        )
    )


def doc_to_target(doc):
    return doc["tgt"]
