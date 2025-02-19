from collections.abc import Iterable
import numpy as np
import evaluate
from moverscore_v2 import get_idf_dict, word_mover_score


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt", "bleurt-base-512", module_type="metric")


def doc_to_text(doc) -> str:
    text = doc["CHQ"]
    idx = text.find('MESSAGE')
    if idx != -1:
        return text[idx+9:]
    else:
        return text


def doc_to_target(doc) -> str:
    return doc["Summary"]


def process_results_gen(doc, results):
    pred, refs = [results[0]], [doc_to_target(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {
            "BLEU": np.NAN,
            "ROUGE-1": np.NAN,
            "ROUGE-2": np.NAN,
            "ROUGE-L": np.NAN,
            "BERT": np.NAN,
            "BLEURT": np.NAN,
            "MOVERSCORE": np.NAN,
        }

    hyp, ref = [pred[0], ''], [refs[0], '']
    idf_dict_hyp = get_idf_dict(hyp)  # idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = get_idf_dict(ref)  # idf_dict_ref = defaultdict(lambda: 1.)

    moverscore_results = word_mover_score(ref, hyp, idf_dict_ref, idf_dict_hyp,
                                          stop_words=[], n_gram=1, remove_subwords=False)[0]

    try:
        bleu_results = bleu.compute(predictions=pred, references=refs)
        rouge_results = rouge.compute(predictions=pred, references=refs)
        bert_results = bertscore.compute(predictions=pred, references=refs, lang="en")
        bleurt_results = bleurt.compute(predictions=pred, references=refs)
    except:
        bleu_results = {"bleu": np.NAN}
        bleurt_results = {"scores": np.NAN}
        rouge_results = {"rouge1": np.NAN, "rouge2": np.NAN, "rougeL": np.NAN}
        bert_results = {"f1": np.NAN}

    return {
        "BLEU": bleu_results["bleu"],
        "ROUGE-1": rouge_results["rouge1"],
        "ROUGE-2": rouge_results["rouge2"],
        "ROUGE-L": rouge_results["rougeL"],
        "BERT": np.nanmean(bert_results["f1"]),
        "BLEURT": np.nanmean(bleurt_results["scores"]),
        "MOVERSCORE": moverscore_results,
    }
