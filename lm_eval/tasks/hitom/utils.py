import re


def _parse_doc(doc):
    # choices arrives as "A. foo, B. bar, ..."; answer is the text value, not the letter
    choices = [c.split(". ", 1)[1].strip() for c in doc["choices"].split(", ")]
    return {**doc, "choices_list": choices, "gold": choices.index(doc["answer"])}


def process_docs(dataset):
    return dataset.map(_parse_doc)


def _make_filter(style, order):
    def _process(dataset):
        return dataset.filter(
            lambda d: d["prompting_type"] == style and d["question_order"] == order
        ).map(_parse_doc)

    return _process


# Per (prompting_type x question_order) doc processors, e.g. process_docs_cotp_order_0
for _style in ("CoTP", "VP"):
    for _order in range(5):
        globals()["process_docs_%s_order_%d" % (_style.lower(), _order)] = _make_filter(
            _style, _order
        )


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
