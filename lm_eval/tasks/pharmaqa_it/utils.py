import re
import string
from collections import Counter


ARTICLES = r"\b(il|lo|la|i|gli|le|un|uno|una|l|un)\b"


def doc_to_text(doc):
    return (
        f"Documento:\n{doc['document_text']}\n\nDomanda: {doc['question']}\nRisposta:"
    )


def _norm(text):
    text = text.lower().replace("’", "'")
    text = "".join(c if c not in string.punctuation else " " for c in text)
    text = re.sub(ARTICLES, " ", text)
    return " ".join(text.split())


def process_results(doc, results):
    pred = _norm(results[0].strip().split("\n")[0])
    gold = _norm(doc["answer"])
    p, g = pred.split(), gold.split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g:
        f1 = float(p == g)
    elif common == 0:
        f1 = 0.0
    else:
        prec, rec = common / len(p), common / len(g)
        f1 = 2 * prec * rec / (prec + rec)
    return {"exact_match": float(pred == gold), "f1": f1}
