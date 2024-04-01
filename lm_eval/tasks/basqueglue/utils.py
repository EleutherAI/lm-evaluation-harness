import html
import re

from datasets import load_metric


def general_detokenize(string):
    string = re.sub(r"\s+([.,;:!?)])", r"\1", string)
    string = re.sub(r"(\s+|^)\(\s+([^)]+)\s+\)", r"\1(\2)", string)
    string = re.sub(r"(\s+|^)\[\s+([^)]+)\s+\]", r"\1[\2]", string)
    string = re.sub(r'(\s+|^)"\s+([^"]+)\s+"', r'\1"\2"', string)
    string = re.sub(r"(\s+|^)'\s+([^']+)\s+'", r"\1'\2'", string)
    return string


def process_doc(string):
    string = html.unescape(string)
    string = general_detokenize(string)
    return string


def process_wic_docs(dataset):
    def _helper(doc):
        # there's some issues with the encoding on this one
        doc["sentence1"] = (
            process_doc(doc["sentence1"]).encode("latin-1").decode("utf-8")
        )
        doc["sentence2"] = (
            process_doc(doc["sentence2"]).encode("latin-1").decode("utf-8")
        )
        return doc

    return dataset.map(_helper)


def coref_doc_to_text(x):
    def _span_in_context(span_index, span_text):
        span_start = span_index
        span_end = span_start + len(span_text.split(" ")) - 1
        tokens[span_start] = f"*{tokens[span_start]}"
        tokens[span_end] = f"{tokens[span_end]}*"

    tokens = x["text"].split(" ")
    _span_in_context(x["span1_index"], x["span1_text"])
    _span_in_context(
        x["span2_index"] - 1, x["span2_text"]
    )  # span1_index is 0-based but span2_index is 1-based ??
    context = process_doc(" ".join(tokens))
    span_1 = process_doc(x["span1_text"])
    span_2 = process_doc(x["span2_text"])
    text = (
        f"Testua: {context}\n"
        + f'Galdera: Aurreko testuan, "*{span_1}*" eta "*{span_2}*" gauza bera dira?\n'
        + "Erantzuna:"
    )
    return text


# Measure F1 as in the benchmark repo: https://github.com/orai-nlp/BasqueGLUE/blob/main/eval_basqueglue.py


def micro_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="micro")[
        "f1"
    ]
    return f1_score


def vaxx_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_class = f1_metric.compute(
        references=golds, predictions=preds, labels=[0, 2], average=None
    )["f1"]
    f1_score = sum(f1_class) / len(f1_class)
    return f1_score
