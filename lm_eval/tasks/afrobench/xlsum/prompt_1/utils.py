import unicodedata

import evaluate


def _unicode_tokenize(text):
    """Lowercase and split on anything that is not a letter, mark or number.

    rouge_score's default tokenizer keeps only ``[a-z0-9]``, so text in
    Ge'ez (Amharic, Tigrinya), Arabic or Telugu script is dropped entirely
    and ROUGE is 0 even for an exact match. Keeping Unicode letters (L*),
    combining marks (M*, e.g. Telugu vowel signs, Yoruba tone marks) and
    numbers (N*) fixes that while producing identical tokens for ASCII text.
    """
    return "".join(
        c if unicodedata.category(c)[0] in "LMN" else " " for c in text.lower()
    ).split()


def rougeL(items):
    """
    # passthrough for efficiency
    """
    return items


def rougeL_agg(items):
    """
    Higher is better
    """
    refs, preds = zip(*items, strict=True)
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(
        predictions=preds, references=refs, tokenizer=_unicode_tokenize
    )["rougeL"]
