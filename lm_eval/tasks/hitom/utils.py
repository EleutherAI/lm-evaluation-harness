import re
from pathlib import Path

import datasets


def _data_file():
    # Resolve THIS repo's own Hi-ToM submodule copy, relative to this file, so the
    # task is portable across clones/machines (no hardcoded absolute path).
    for parent in Path(__file__).resolve().parents:
        cand = parent / "benchmarks/Hi-ToM_dataset/Hi-ToM_data/Hi-ToM_data.json"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "Hi-ToM_data.json not found under any parent's benchmarks/ directory"
    )


def _parse_doc(doc):
    # choices arrives as "A. foo, B. bar, ..."; answer is the text value, not the letter.
    # Contract: answer appears verbatim among the choice values, else .index raises (fail loud).
    choices = [c.split(". ", 1)[1].strip() for c in doc["choices"].split(", ")]
    return {**doc, "choices_list": choices, "gold": choices.index(doc["answer"])}


def load(prompting_type=None, question_order=None, **kwargs):
    """LOADER: read the verbatim Hi-ToM JSON, parse choices/gold, and optionally
    filter to one (prompting_type, question_order) partition. The partition
    selectors arrive from each leaf task's `dataset_kwargs`; with neither set this
    returns the whole corpus."""
    ds = datasets.load_dataset(
        "json", data_files=_data_file(), field="data", split="train"
    ).map(_parse_doc)
    if prompting_type is not None:
        ds = ds.filter(lambda d: d["prompting_type"] == prompting_type)
    if question_order is not None:
        ds = ds.filter(lambda d: d["question_order"] == question_order)
    return {"train": ds}


def _extract_index(gen, choices):
    """Letter-first, value-fallback extraction (Hi-ToM answers come first in the output)."""
    n = len(choices)
    last = chr(ord("A") + n - 1)
    lc = "A-%sa-%s" % (last, last.lower())
    letter_patterns = [
        r"^\s*\(?([%s])[\.\):]" % lc,            # leading "A." "A)" "A:" "(A)"
        r"^\s*\(?([%s])\)?\s*$" % lc,            # whole (stripped) output is just a letter
        r"answer\s*(?:is|:)\s*\(?([%s])\b" % lc,  # "the answer is A" / "answer: A"
        r"\(([%s])\)" % lc,                       # "(A)" anywhere
    ]
    g = gen.strip()
    for pat in letter_patterns:
        m = re.search(pat, g, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            idx = ord(m.group(1).upper()) - ord("A")
            if 0 <= idx < n:
                return idx
    # fallback: earliest-appearing choice value (case-insensitive)
    low = gen.lower()
    best = None
    for i, v in enumerate(choices):
        pos = low.find(v.lower())
        if pos != -1 and (best is None or pos < best[0]):
            best = (pos, i)
    return best[1] if best else None


def process_results(doc, results):
    pred = _extract_index(results[0], doc["choices_list"])
    return {"acc": 1.0 if pred == doc["gold"] else 0.0}
