import datasets


LABELS = {
    "English": (
        "Passage",
        "Question",
        "Answer",
        "Write a short Socratic hint that helps a student find the answer without revealing it.\nHint",
    ),
    "Italian": (
        "Testo",
        "Domanda",
        "Risposta",
        "Scrivi un breve suggerimento socratico che aiuti lo studente a trovare la risposta senza rivelarla.\nSuggerimento",
    ),
    "Farsi": (
        "متن",
        "پرسش",
        "پاسخ",
        "یک راهنمایی کوتاه سقراطی بنویس که به دانش‌آموز کمک کند پاسخ را پیدا کند، بدون اینکه پاسخ را لو بدهد.\nراهنمایی",
    ),
}


def _subset(lang):
    def _process(dataset: datasets.Dataset) -> datasets.Dataset:
        return dataset.filter(lambda d: d["language"] == lang)

    return _process


process_en = _subset("English")
process_it = _subset("Italian")
process_fa = _subset("Farsi")


def doc_to_text(doc):
    p, q, a, h = LABELS[doc["language"]]
    return f"{p}: {doc['passage']}\n{q}: {doc['question']}\n{a}: {doc['answer']}\n{h}:"
