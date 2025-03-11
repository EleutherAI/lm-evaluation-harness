import numpy as np


try:
    import evaluate

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleurt = evaluate.load("bleurt", "bleurt-base-512", module_type="metric")

except (ModuleNotFoundError, ImportError):
    raise ModuleNotFoundError(
        "Please install evaluation metrics via pip install evaluate and pip install bert-score",
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
        bleu_results = {"bleu": np.NAN}

    try:
        rouge_results = rouge.compute(predictions=pred, references=refs)
    except Exception as e:
        print(f"Rouge error: {e}")
        rouge_results = {"rouge1": np.NAN, "rouge2": np.NAN, "rougeL": np.NAN}

    try:
        bleurt_scores = bleurt.compute(predictions=pred, references=refs)["scores"]
    except Exception as e:
        print(f"Bleurt error: {e}")
        bleurt_scores = [np.NAN]

    try:
        bert_scores = bertscore.compute(predictions=pred, references=refs, lang="en")[
            "f1"
        ]
    except Exception as e:
        print(f"Bert error: {e}")
        bert_scores = [np.NAN]

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


def doc_to_text(doc) -> str:
    return doc["dialogue"]


def doc_to_target(doc) -> str:
    return doc["section_text"]


def process_results(doc, results):
    pred, refs = [results[0]], [doc_to_target(doc)]

    if len(refs[0]) < 5 or len(pred[0]) < 5:
        return {
            "bleu": np.NAN,
            "rouge1": np.NAN,
            "rouge2": np.NAN,
            "rougeL": np.NAN,
            "bleurt": np.NAN,
            "bert_score": np.NAN,
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
