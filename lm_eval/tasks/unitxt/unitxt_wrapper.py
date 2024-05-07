try:
    from unitxt import evaluate
except ImportError:
    raise ImportError(
        "Package 'unitxt' is not installed. To install it, use `pip install 'lm_eval[unitxt]'`"
    )

from lm_eval.api.registry import AGGREGATION_REGISTRY, METRIC_REGISTRY, register_metric


def unitxt_agg_metric(items):
    preds = [pred[0] for pred, _, _ in items]
    refs = [ref for _, ref, _ in items]
    metric_name = items[0][2].replace("unitxt_", "metrics.")
    for ref in refs:
        ref["metrics"] = [metric_name]

    result_metrics = evaluate(preds, refs)
    return result_metrics[0]["score"]["global"]["score"]


AGGREGATION_REGISTRY["unitxt"] = unitxt_agg_metric


def unitxt_metric(items):  # This is a passthrough function
    return items


def process_results(doc, results):
    metrics = doc["metrics"]
    scores = {}
    for metric in metrics:
        metric = metric.replace("metrics.", "unitxt_")
        scores[metric] = (results, doc, metric)

        if metric not in METRIC_REGISTRY:
            register_metric(
                metric=metric,
                higher_is_better=True,
                output_type="generate_until",
                aggregation="unitxt",
            )(unitxt_metric)
    return scores


#
