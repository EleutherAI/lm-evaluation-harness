import numpy as np


try:
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", "bleurt-base-512", module_type="metric")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via pip install evaluate bert-score "
        "rouge_score>=0.1.2 nltk absl-py "
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


def doc_to_text_raw(doc) -> str:
    return doc["description"]


def doc_to_target_raw(doc) -> str:
    return doc["utterances"]["utterance"][1]


def process_results_gen_raw(doc, results):
    pred, refs = [results[0]], [doc_to_target_raw(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bleurt": np.nan,
            "bert_score": np.nan,
        }

    results = doc_eval(pred, refs)

    return {
        "bleu": results["bleu"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "bleurt": results["bleurt"],
        "bert_score": results["bert_score"],
    }


def doc_to_text_qsumm(doc) -> str:
    return doc["src"]


def doc_to_target_qsumm(doc) -> str:
    return doc["tgt"]


def process_results_gen_qsumm(doc, results):
    pred, refs = [results[0]], [doc_to_target_qsumm(doc)]

    if len(refs[0]) < 1 or len(pred[0]) < 1:
        return {
            "bleu": np.nan,
            "rouge1": np.nan,
            "rouge2": np.nan,
            "rougeL": np.nan,
            "bleurt": np.nan,
            "bert_score": np.nan,
        }

    results = doc_eval(pred, refs)

    return {
        "bleu": results["bleu"],
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "bleurt": results["bleurt"],
        "bert_score": results["bert_score"],
    }
