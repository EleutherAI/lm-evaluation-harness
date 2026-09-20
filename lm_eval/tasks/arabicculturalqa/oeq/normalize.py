"""Paper-matching normalization for ArabicCulturalQA OEQ.

Ports the Arabic + English normalization from the project's `qa_eval.py` so
that lm-eval-harness BERTScore / ROUGE-L numbers line up with the paper.

Soft deps:
- `camel-tools` (Arabic): dediacritization + alef/hamza/teh-marbuta unification.
  Without it we still strip diacritics and do hamza unification, which catches
  most of the variance.
- `nltk` (English): WordNet lemmatization + Porter stemming. Without it we
  lowercase / strip URLs / strip punctuation and leave the surface forms alone.

Both libs are loaded lazily; the normalizer downgrades silently if either is
missing.
"""

from __future__ import annotations

import re
import string
from functools import lru_cache


PUNCTS = string.punctuation

DISFLUENCIES = [
    "اه",
    "آه",
    "آ",
    "ااا",
    "أه",
    "م",
    "آآ",
    "ااه",
    "إمم",
    "أوه",
    "وال",
    "اااه",
    "اااا",
    "ااااا",
    "ااااه",
    "غير_واضح",
]

DROP_PATTERNS = [
    r"#overlap#",
    r"@music@",
    r"@noise@",
    r"@no_sound@",
    r"@laugh@",
    r"@clap@",
    r"##not_clear##",
    r"\bS1\b",
    r"\bS2\b",
]


# ---- camel-tools (soft) ----------------------------------------------------


@lru_cache(maxsize=1)
def _camel():
    try:
        from camel_tools.utils.charsets import AR_LETTERS_CHARSET
        from camel_tools.utils.dediac import dediac_ar
        from camel_tools.utils.normalize import (
            normalize_alef_ar,
            normalize_alef_maksura_ar,
            normalize_teh_marbuta_ar,
        )

        return {
            "dediac_ar": dediac_ar,
            "alef_ar": normalize_alef_ar,
            "alef_maksura_ar": normalize_alef_maksura_ar,
            "teh_marbuta_ar": normalize_teh_marbuta_ar,
            "letters": AR_LETTERS_CHARSET,
        }
    except ImportError:
        return None


# ---- NLTK (soft) -----------------------------------------------------------

_NLTK_READY = False


@lru_cache(maxsize=1)
def _nltk():
    global _NLTK_READY
    try:
        import nltk
        from nltk.stem import PorterStemmer, WordNetLemmatizer

        if not _NLTK_READY:
            for pkg in ("wordnet", "omw-1.4"):
                # Best-effort download; offline environments fall back to the
                # surface-form pipeline below.
                try:
                    nltk.download(pkg, quiet=True)
                except (OSError, LookupError):
                    pass
            _NLTK_READY = True
        return WordNetLemmatizer(), PorterStemmer()
    except ImportError:
        return None


# ---- small helpers ---------------------------------------------------------


def _normalize_digits(text: str) -> str:
    digit_map = str.maketrans(
        {
            "٠": "0",
            "١": "1",
            "٢": "2",
            "٣": "3",
            "٤": "4",
            "٥": "5",
            "٦": "6",
            "٧": "7",
            "٨": "8",
            "٩": "9",
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
        }
    )
    return text.translate(digit_map)


def _normalize_hamza_ar(text: str) -> str:
    return text.replace("إ", "ا").replace("أ", "ا").replace("ؤ", "ء").replace("ئ", "ء")


def _rm_consecutive_duplicates(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    out = [words[0]]
    for w in words[1:]:
        if w != out[-1]:
            out.append(w)
    return " ".join(out)


def _remove_disfluencies(text: str) -> str:
    pattern = r"\b(?:" + "|".join(map(re.escape, DISFLUENCIES)) + r")\b"
    text = re.sub(pattern, "", text)
    return re.sub(r"\s+", " ", text).strip()


def _remove_punctuations(text: str) -> str:
    return text.translate(str.maketrans("", "", PUNCTS))


# ---- public API ------------------------------------------------------------


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = _normalize_digits(text)
    for pat in DROP_PATTERNS:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    cam = _camel()
    if cam:
        text = cam["alef_maksura_ar"](text)
        text = _normalize_hamza_ar(text)
        text = cam["dediac_ar"](text)
        text = cam["alef_ar"](text)
        text = cam["teh_marbuta_ar"](text)
    else:
        text = _normalize_hamza_ar(text)

    text = text.replace(".", "").replace("▁", " ").lower()

    if cam and cam["letters"]:
        keep = cam["letters"].union(
            set(string.ascii_letters + string.digits + string.whitespace),
            {"ﷺ", "ﷻ", "ﷲ", "ﷴ"},
        )
        chars = set("".join(text.split()))
        for ch in chars.difference(keep):
            text = text.replace(ch, "")
    else:
        text = re.sub(r"[^؀-ۿa-zA-Z0-9\s]", "", text)

    text = _rm_consecutive_duplicates(text)
    text = _remove_disfluencies(text)
    text = _remove_punctuations(text)
    return " ".join(text.split())


def normalize_english(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = text.translate(str.maketrans("", "", PUNCTS))
    text = re.sub(r"\s+", " ", text)

    nltk_pair = _nltk()
    if nltk_pair is None:
        return text.strip()
    lemmatizer, stemmer = nltk_pair
    return " ".join(stemmer.stem(lemmatizer.lemmatize(w)) for w in text.split()).strip()


def normalize(text: str, lang: str) -> str:
    return normalize_arabic(text) if lang == "ar" else normalize_english(text)
