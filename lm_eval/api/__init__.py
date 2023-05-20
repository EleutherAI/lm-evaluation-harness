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
    "median": metrics.median,
    "perplexity": metrics.perplexity,
}

HIGHER_IS_BETTER_REGISTRY = {
    "matthews_corrcoef": True,
    "f1_score": True,
    "perplexity": False,
    "bleu": True,
    "chrf": True,
    "ter": False,

    "acc": True,
    "acc_norm": True,
    "acc_mutual_info": True,
    "word_perplexity": False,
    "byte_perplexity": False,
    "bits_per_byte": False,
}