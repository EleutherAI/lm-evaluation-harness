import datasets
import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics
from lm_eval.api.metrics import metric_max_over_ground_truths

def doc_to_target(doc):
    return doc["answers"]["text"][0]

def filter_arabic(dataset):
    return dataset.filter(lambda example: example["id"].startswith("arabic"))

def filter_bengali(dataset):
    return dataset.filter(lambda example: example["id"].startswith("bengali"))

def filter_finnish(dataset):
    return dataset.filter(lambda example: example["id"].startswith("finnish"))

def filter_indonesian(dataset):
    return dataset.filter(lambda example: example["id"].startswith("indonesian"))

def filter_russian(dataset):
    return dataset.filter(lambda example: example["id"].startswith("russian"))

def filter_korean(dataset):
    return dataset.filter(lambda example: example["id"].startswith("korean"))

def filter_english(dataset):
    return dataset.filter(lambda example: example["id"].startswith("english"))

def filter_swahili(dataset):
    return dataset.filter(lambda example: example["id"].startswith("swahili"))

def filter_telugu(dataset):
    return dataset.filter(lambda example: example["id"].startswith("telugu"))

def process_results(doc, results):
    gold_label_set = doc["answers"]["text"]
    f1 = metric_max_over_ground_truths(
        squad_metrics.compute_f1, results[0][0], gold_label_set
    )
    em = metric_max_over_ground_truths(
        squad_metrics.compute_exact, results[0][0], gold_label_set
    )

    return {
        "f1": f1,
        "em": em,
    }
