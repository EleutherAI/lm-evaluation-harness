import unicodedata

import evaluate


def strip(resps, docs):
    """
    Assuming each entry of `resps` is a list of model responses, we discard all but the first response.
    """
    return (r[0].strip() for r in resps)


def doc_to_text(doc):
    doc_text = doc["messages"][0]["content"].replace(
        "لخص هاد المقطع", "لخص هاد المقطع في ٣٠ كلمة"
    )
    return doc_text


def doc_to_target(doc):
    return doc["messages"][1]["content"]


def bert(items):
    return items


def Average(lst):
    return sum(lst) / len(lst)


def darijabert(items):
    bert_model = "SI2M-Lab/DarijaBERT"
    bert_score = evaluate.load("bertscore")
    predictions, references = zip(*items, strict=False)
    bert = bert_score.compute(
        predictions=predictions,
        references=references,
        model_type=bert_model,
        num_layers=12,
    )
    return Average(bert["f1"])


def rouge1(items):
    return items


def rougeL(items):
    return items


def rouge2(items):
    return items


def rougeLsum(items):
    return items


def _rouge_tokenize(text):
    """Split Unicode words while retaining Arabic letters and combining marks."""
    return "".join(
        char if unicodedata.category(char)[0] in "LMN" else " " for char in text.lower()
    ).split()


def agg_rougelsum(items):
    rouge = evaluate.load("rouge")
    predictions, references = zip(*items, strict=False)
    return rouge.compute(
        predictions=predictions, references=references, tokenizer=_rouge_tokenize
    )["rougeLsum"]


def agg_rouge1(items):
    rouge = evaluate.load("rouge")
    predictions, references = zip(*items, strict=False)
    return rouge.compute(
        predictions=predictions, references=references, tokenizer=_rouge_tokenize
    )["rouge1"]


def agg_rouge2(items):
    rouge = evaluate.load("rouge")
    predictions, references = zip(*items, strict=False)
    return rouge.compute(
        predictions=predictions, references=references, tokenizer=_rouge_tokenize
    )["rouge2"]


def agg_rougel(items):
    rouge = evaluate.load("rouge")
    predictions, references = zip(*items, strict=False)
    return rouge.compute(
        predictions=predictions, references=references, tokenizer=_rouge_tokenize
    )["rougeL"]
