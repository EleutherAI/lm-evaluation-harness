"""Prompts and scoring for serbeval (Serbian grammar, morphology and orthography).

Self-contained port of the scoring used by https://github.com/djolex999/serbeval, so the
numbers here match that project's leaderboard. Three item shapes:

* MCQ: A/B/C/D, scored on the first standalone letter in the response.
* Cloze: the model writes one inflected form. Case, outer punctuation and script
  (Cyrillic/Latin) are folded; Serbian diacritics and intra-word hyphens are NOT
  ("kuca" != "kuća", "kontakt osoba" != "kontakt-osoba").
* Correction: the model returns a sentence corrected (or unchanged) after "Ispravljeno:".

Every task also reports ``unparsed``: the share of responses with nothing readable in
them, so a formatting failure is not mistaken for a language error.
"""

import re
import unicodedata
from collections.abc import Sequence


LETTERS = ("A", "B", "C", "D")
BLANK = "____"
_LETTER = re.compile(r"\b([ABCD])\b")

# --- MCQ ----------------------------------------------------------------------


def doc_to_text(doc: dict) -> str:
    lines = [doc["question"], ""]
    for letter, choice in zip(LETTERS, doc["choices"], strict=False):
        lines.append(f"{letter}) {choice}")
    lines.append("")
    lines.append("Odgovori samo slovom tačnog odgovora (A, B, C ili D).")
    lines.append("Odgovor:")
    return "\n".join(lines)


def doc_to_target(doc: dict) -> str:
    return LETTERS[doc["answer_index"]]


def _letter(raw: str) -> str | None:
    # Case-sensitive on purpose: the Serbian conjunctions "a" and "i" would otherwise read
    # as answer letters when a model explains before answering.
    match = _LETTER.search(raw or "")
    return match.group(1) if match else None


def _mcq_scores(doc: dict, results: list[str]) -> dict:
    pred = _letter(results[0]) if results else None
    return {
        "exact_match": 1.0 if pred == LETTERS[doc["answer_index"]] else 0.0,
        "unparsed": 0.0 if pred is not None else 1.0,
    }


def process_results_mcq(doc: dict, results: list[str]) -> dict:
    return _mcq_scores(doc, results)


def process_results_pair(doc: dict, results: list[str]) -> dict:
    scores = _mcq_scores(doc, results)
    scores["pair_agreement"] = (doc["pair_id"], scores["exact_match"])
    return scores


def pair_agreement(items: Sequence[tuple[str, float]]) -> float:
    """Share of Cyrillic/Latin pairs where both scripts got the same outcome."""
    by_pair: dict[str, list[float]] = {}
    for pair_id, correct in items:
        by_pair.setdefault(pair_id, []).append(correct)
    complete = [vals for vals in by_pair.values() if len(vals) == 2]
    if not complete:
        return 0.0
    return sum(1 for vals in complete if vals[0] == vals[1]) / len(complete)


# --- shared normalization -------------------------------------------------------------

# Cyrillic -> Latin is unambiguous for Serbian (each Cyrillic letter has one Latin spelling).
_CYR = "АБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШабвгдђежзијклљмнњопрстћуфхцчџш"
_LAT = [
    "A",
    "B",
    "V",
    "G",
    "D",
    "Đ",
    "E",
    "Ž",
    "Z",
    "I",
    "J",
    "K",
    "L",
    "Lj",
    "M",
    "N",
    "Nj",
    "O",
    "P",
    "R",
    "S",
    "T",
    "Ć",
    "U",
    "F",
    "H",
    "C",
    "Č",
    "Dž",
    "Š",
    "a",
    "b",
    "v",
    "g",
    "d",
    "đ",
    "e",
    "ž",
    "z",
    "i",
    "j",
    "k",
    "l",
    "lj",
    "m",
    "n",
    "nj",
    "o",
    "p",
    "r",
    "s",
    "t",
    "ć",
    "u",
    "f",
    "h",
    "c",
    "č",
    "dž",
    "š",
]
_TO_LATIN = dict(zip(_CYR, _LAT, strict=True))

_INTRA_WORD_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")
_KEPT_HYPHEN = "\x00"


def _normalize(text: str) -> str:
    """Casefold, strip punctuation/whitespace; keep diacritics and intra-word hyphens."""
    protected = _INTRA_WORD_HYPHEN.sub(_KEPT_HYPHEN, text)
    stripped = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in protected
    )
    return " ".join(stripped.lower().split()).replace(_KEPT_HYPHEN, "-")


def _comparable(text: str) -> str:
    return _normalize("".join(_TO_LATIN.get(ch, ch) for ch in text))


# --- cloze -------------------------------------------------------------------------------

_PREFIXES = re.compile(
    r"^\s*(odgovor|rešenje|resenje|answer|tačan oblik|oblik)\s*[:\-—]\s*",
    re.IGNORECASE,
)
_WRAPPERS = "\"'`„“”‘’«»()[]{}*_ \t.,;:!?"
_MAX_ANSWER_WORDS = 3


def doc_to_text_cloze(doc: dict) -> str:
    return (
        "Dopuni rečenicu pravilnim oblikom reči u zagradi.\n\n"
        f"{doc['question']}  ({doc['cue']})\n\n"
        "Odgovori samo traženim oblikom reči, bez objašnjenja.\n"
        "Odgovor:"
    )


def doc_to_target_cloze(doc: dict) -> str:
    return doc["answers"][0]


def _extract_form(raw: str) -> str:
    """First non-empty line short enough to be an answer, else the first line."""
    lines = [
        cleaned
        for line in (raw or "").splitlines()
        if (cleaned := _PREFIXES.sub("", line.strip()).strip(_WRAPPERS))
    ]
    if not lines:
        return ""
    for line in lines:
        if len(line.split()) <= _MAX_ANSWER_WORDS:
            return line
    return lines[0]


def _slot_from_sentence(raw: str, question: str) -> str | None:
    """The form at the blank when the response repeats the whole sentence verbatim."""
    raw_head, _, raw_tail = question.partition(BLANK)
    head, tail = _comparable(raw_head), _comparable(raw_tail)
    for line in (raw or "").splitlines():
        text = _comparable(_PREFIXES.sub("", line))
        if not (text.startswith(head) and text.endswith(tail)):
            continue
        middle = text[len(head) : len(text) - len(tail)]
        if (head and not middle.startswith(" ")) or (tail and not middle.endswith(" ")):
            continue
        slot = middle.strip()
        if slot and len(slot.split()) <= _MAX_ANSWER_WORDS:
            return slot
    return None


def process_results_cloze(doc: dict, results: list[str]) -> dict:
    raw = results[0] if results else ""
    answers = [_comparable(a) for a in doc["answers"]]
    produced = _comparable(_extract_form(raw))
    correct = bool(produced) and produced in answers
    if not correct:
        slot = _slot_from_sentence(raw, doc["question"])
        correct = slot is not None and slot in answers
    return {
        "exact_match": 1.0 if correct else 0.0,
        "unparsed": 0.0 if _extract_form(raw) else 1.0,
    }


# --- correction ------------------------------------------------------------------------

_MARKER = re.compile(r"ispravljeno\s*:\s*", re.IGNORECASE)
_CORRECTION_WRAPPERS = "\"'`„“”‘’«»*_ \t"


def doc_to_text_correction(doc: dict) -> str:
    return (
        "Sledeća rečenica možda sadrži jednu pravopisnu grešku, a možda je potpuno "
        "ispravna. Ako ima grešku, ispravi je i ne menjaj ništa drugo. Ako je ispravna, "
        "prepiši je bez izmena.\n\n"
        f"Rečenica: {doc['text']}\n\n"
        "Odgovori jednim redom u obliku:\n"
        "Ispravljeno: <rečenica>"
    )


def doc_to_target_correction(doc: dict) -> str:
    return f"Ispravljeno: {doc['answers'][0]}"


def _extract_sentence(raw: str) -> str:
    """The sentence after the last "Ispravljeno:", else the first non-empty line."""
    text = raw or ""
    markers = list(_MARKER.finditer(text))
    if markers:
        rest = text[markers[-1].end() :].strip().splitlines()
        return rest[0].strip(_CORRECTION_WRAPPERS) if rest else ""
    for line in text.splitlines():
        if line.strip():
            return line.strip().strip(_CORRECTION_WRAPPERS)
    return ""


def process_results_correction(doc: dict, results: list[str]) -> dict:
    raw = results[0] if results else ""
    sentence = _extract_sentence(raw)
    produced = _comparable(sentence)
    correct = bool(produced) and any(produced == _comparable(a) for a in doc["answers"])
    return {
        "exact_match": 1.0 if correct else 0.0,
        "unparsed": 0.0 if sentence else 1.0,
    }
