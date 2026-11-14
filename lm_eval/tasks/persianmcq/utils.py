import re

import datasets


LETTERS = ["الف", "ب", "ج", "د"]
RIGHT, WRONG = "(پاسخ صحیح)", "(پاسخ غلط)"


def parse_mcq(text):
    """Split a refined PersianMCQ item into question, four options and the gold index."""
    text = text.replace("\u200f", "").strip()
    pos, cut = 0, []
    for letter in LETTERS:
        m = re.compile(rf"(?:^|[\s؟?.:)]){letter}\s*[.)]\s*").search(text, pos)
        if not m:
            return None
        cut.append((m.start(), m.end()))
        pos = m.end()
    question = (
        text[: cut[0][0] + 1].strip()
        if text[cut[0][0]] in "؟?.:)"
        else text[: cut[0][0]].strip()
    )
    options, gold = [], []
    for i, (_, e) in enumerate(cut):
        end = (
            cut[i + 1][0] + (1 if text[cut[i + 1][0]] in "؟?.:)" else 0)
            if i + 1 < 4
            else len(text)
        )
        opt = text[e:end].strip()
        if RIGHT in opt:
            gold.append(i)
        opt = opt.replace(RIGHT, "").replace(WRONG, "").strip().rstrip("،,;").strip()
        options.append(opt)
    if len(gold) != 1 or not question or not all(options):
        return None
    return question, options, gold[0]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _add(doc):
        parsed = parse_mcq(doc["MCQs Refinement"] or "")
        q, o, g = parsed or ("", ["", "", "", ""], -1)
        return {"question": q, "choices": o, "gold": g}

    return dataset.map(_add).filter(lambda d: d["gold"] >= 0)


def doc_to_text(doc):
    return f"متن: {doc['Augmented Text']}\nپرسش: {doc['question']}\nپاسخ:"
