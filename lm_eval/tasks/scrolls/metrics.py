import evaluate

rouge_fn = evaluate.load('rouge')

def rouge1(predictions, references):
    results = rouge_fn.compute(predictions=predictions, references=references)
    return results['rouge1']

def rouge2(predictions, references):
    results = rouge_fn.compute(predictions=predictions, references=references)
    return results['rouge2']

def rougeL(predictions, references):
    results = rouge_fn.compute(predictions=predictions, references=references)
    return results['rougeL']

squad_metric = evaluate.load("squad_v2")

def agg_f1(samples):
    predictions, references = zip(*samples)  # unzip, if you will
    computed = squad_metric.compute(predictions=predictions, references=references)
    return computed["f1"]


def _download_metric():
    import os
    import shutil
    from huggingface_hub import hf_hub_download
    scrolls_metric_path = hf_hub_download(repo_id="tau/scrolls", repo_type="dataset", filename="metrics/scrolls.py")
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path) + os.path.basename(scrolls_metric_path).replace(".", "_") + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path