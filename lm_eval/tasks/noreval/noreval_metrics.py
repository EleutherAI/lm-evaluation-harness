"""Shared scoring for the NorEval multiple-choice tasks.

A custom process_results replaces lm-eval's built-in multiple_choice scoring
entirely, so the accuracies are re-implemented here with the harness's
formulas, alongside their continuous counterparts.  Each task's yaml points
doc_to_choice at the same function it passes to `make_process_results`, so the
strings the harness scores are the strings normalized here.

Scores come in three variants: raw loglikelihoods (""), per-character
loglikelihoods ("_norm"; a ranking no-op when all choices have equal length),
and PMI ("_mutual_info"; conditional minus unconditional loglikelihoods,
identically zero in zero-shot runs of tasks with an empty prompt).  Each
variant yields:
  * "acc*":          1 if the argmax is a correct choice,
  * "prob_correct*": softmax mass on the correct choice(s) -- for "_norm"
                     this is a temperature-scaled distribution (temperature
                     ~ choice length), not a calibrated probability, and
  * "loglikelihood_correct*": the best correct choice's score, passed through.
"""

from scipy.special import softmax


def make_process_results(
    choices, gold_indices, postprocess=None, expect_mutual_info=True
):
    """Build a task's process_results.

    choices(doc) returns the answer choices and must be the same function the
    task's doc_to_choice references; gold_indices(doc) returns the indices of
    the correct choices.  The optional postprocess(metrics, predictions, doc)
    can add entries such as (prediction, gold) pairs for F1.
    expect_mutual_info declares whether "_mutual_info" metrics are in the
    metric list; see split_loglikelihoods.
    """

    def process_results(doc, results):
        metrics, predictions = score_choices(
            results, choices(doc), gold_indices(doc), expect_mutual_info
        )
        if postprocess is not None:
            postprocess(metrics, predictions, doc)
        return metrics

    return process_results


def split_loglikelihoods(results, n_choices, expect_mutual_info):
    """Split the flat result list into conditional and unconditional loglikelihoods.

    The harness appends one unconditional ("", choice) request per choice iff
    the literal metric name "acc_mutual_info" is in metric_list, and the
    doubled length is the only signal that it did.  With
    expect_mutual_info=True, missing unconditional loglikelihoods raise here
    rather than as a missing-metric error at aggregation time.
    """
    lls = [
        result[0] if isinstance(result, (tuple, list)) else result for result in results
    ]
    if len(lls) == 2 * n_choices:
        return lls[:n_choices], lls[n_choices:]
    if len(lls) == n_choices:
        if expect_mutual_info:
            raise ValueError(
                "no unconditional loglikelihoods received although a "
                "'*_mutual_info' metric is expected: add the literal name "
                "'acc_mutual_info' to metric_list (or pass "
                "expect_mutual_info=False)"
            )
        return lls, []
    raise ValueError(
        f"Expected {n_choices} or {2 * n_choices} loglikelihoods, got {len(lls)}"
    )


def score_choices(results, choices, gold_indices, expect_mutual_info=True):
    """Score one document.

    Returns (metrics, predictions); predictions maps each variant suffix to
    the predicted choice index, for postprocess hooks such as F1 pairs.
    """
    conditional_lls, unconditional_lls = split_loglikelihoods(
        results, len(choices), expect_mutual_info
    )

    scored_lls = {
        "": conditional_lls,
        "_norm": [
            ll / len(choice)
            for ll, choice in zip(conditional_lls, choices, strict=True)
        ],
    }
    if unconditional_lls:
        scored_lls["_mutual_info"] = [
            conditional - unconditional
            for conditional, unconditional in zip(
                conditional_lls, unconditional_lls, strict=True
            )
        ]

    metrics, predictions = {}, {}
    for suffix, lls in scored_lls.items():
        pred = max(range(len(choices)), key=lambda i: lls[i])
        probs = softmax(lls)
        predictions[suffix] = pred
        metrics[f"acc{suffix}"] = int(pred in gold_indices)
        metrics[f"prob_correct{suffix}"] = float(sum(probs[i] for i in gold_indices))
        metrics[f"loglikelihood_correct{suffix}"] = float(
            max(lls[i] for i in gold_indices)
        )
    return metrics, predictions
