from . import metrics

METRIC_REGISTRY = {
    "matthews_corrcoef": metrics.matthews_corrcoef,
    "f1_score": metrics.f1_score,
    "perplexity": metrics.perplexity,
    "bleu": metrics.bleu,
    "chrf": metrics.chrf,
    "ter": metrics.ter,
}

AGGREGATION_REGISTRY = {
    "mean": metrics.mean,
    "median": metrics.median
}