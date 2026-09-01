import re
from collections.abc import Iterable

import numpy as np


try:
    import evaluate
    from radgraph import F1RadGraph

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", "bleurt-base-512", module_type="metric")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via pip install evaluate bert-score "
        "rouge_score>=0.1.2 nltk absl-py radgraph"
        "git+https://github.com/google-research/bleurt.git"
    )
except Exception as e:
    raise RuntimeError(
        f"Error loading evaluation metrics: {str(e)}. Please check your installation."
    )


def doc_eval(pred, refs):
    try:
        bleu_results = bleu.compute(predictions=pred, references=refs)
    except Exception as e:
        print(f"Bleu error: {e}")
        bleu_results = {"bleu": np.nan}

    try:
        rouge_results = rouge.compute(predictions=pred, references=refs)
    except Exception as e:
        print(f"Rouge error: {e}")
        rouge_results = {"rouge1": np.nan, "rouge2": np.nan, "rougeL": np.nan}

    try:
        bleurt_scores = bleurt.compute(predictions=pred, references=refs)["scores"]
    except Exception as e:
        print(f"Bleurt error: {e}")
        bleurt_scores = [np.nan]

    try:
        bert_scores = bertscore.compute(predictions=pred, references=refs, lang="en")[
            "f1"
        ]
    except Exception as e:
        print(f"Bert error: {e}")
        bert_scores = [np.nan]

    if bleu_results["bleu"] == 0:
        # Sometimes bleu is 0.0 and this breaks the stderr computation.
        bleu_results["bleu"] += 1e-5

    results = {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"],
        "bleurt": np.mean(bleurt_scores),
        "bert_score": np.mean(bert_scores),
    }

    return results


f1radgraph = F1RadGraph(reward_level="partial")


def doc_to_text(doc) -> str:
    text = doc["extractive_notes_summ"]

    a = re.search("IMPRESSION", text, re.IGNORECASE)
    if a is not None:
        a = a.start()
    else:
        a = -1
    b = re.search("FINDING", text, re.IGNORECASE)
    if b is not None:
        b = b.start()
    else:
        b = -1

    if a < b:
        impressions = text[a:b].split("     ")[0]
        findings = text[b:].split("     ")[0]
    else:
        impressions = text[a:].split("     ")[0]
        findings = text[b:a].split("     ")[0]

    if len(findings) < 5 < len(impressions):
        findings = text[:a]

    return "Given the findings: {}.\nSummarize the findings.".format(findings)


def doc_to_target(doc) -> str:
    text = doc["extractive_notes_summ"]

    a = re.search("IMPRESSION", text, re.IGNORECASE)
    if a is not None:
        a = a.start()
    else:
        a = -1
    b = re.search("FINDING", text, re.IGNORECASE)
    if b is not None:
        b = b.start()
    else:
        b = -1

    if a < b:
        impressions = text[a:b].split("     ")[0]
    else:
        impressions = text[a:].split("     ")[0]

    return impressions


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def process_results(doc, results):
    pred, refs = [results[0]], [doc_to_target(doc)]

    if len(refs[0]) < 5 or len(pred[0]) < 5:
        return {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bleurt": np.nan,
            "bert_score": np.nan,
            "F1-Radgraph": np.nan,
        }

    results = doc_eval(pred, refs)

    try:
        radgraph_score, _, _, _ = f1radgraph(hyps=pred, refs=refs)
    except Exception:
        radgraph_score = np.nan

    return {
        "bleu": results["bleu"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "bleurt": results["bleurt"],
        "bert_score": results["bert_score"],
        "F1-Radgraph": radgraph_score,
    }
