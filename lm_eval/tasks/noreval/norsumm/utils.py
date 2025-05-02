import datasets
import numpy as np
from evaluate import load


try:
    import bert_score
    import sacrebleu
    from rouge_score import rouge_scorer, scoring
except ModuleNotFoundError as e:
    raise type(e)(
        "`sacrebleu`, `bert_score`, and `rouge_score` are required for evaluating the model on NorEval."
    ) from e


ROUGE_SCORER = None
BERTSCORE = None


def process_results(doc, results):
    completion = results[0]
    references = doc["summaries"]

    bleu_scores = [bleu([[reference]], [completion]) for reference in references]
    bleu_max = np.nanmax(bleu_scores)
    bleu_avg = np.nanmean(bleu_scores)

    rouge_scores = [rouge([reference], [completion]) for reference in references]
    rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
    rougeL_max = np.nanmax(rougeL_scores)
    rougeL_avg = np.nanmean(rougeL_scores)

    bertscore_f1s = [
        bertscore_f1(references=[reference], predictions=[completion])
        for reference in references
    ]
    bertscore_f1_max = np.nanmax(bertscore_f1s)
    bertscore_f1_avg = np.nanmean(bertscore_f1s)

    return {
        "bleu_max": bleu_max,
        "bleu_avg": bleu_avg,
        "rougeL_max": rougeL_max,
        "rougeL_avg": rougeL_avg,
        "bertscore_f1_max": bertscore_f1_max,
        "bertscore_f1_avg": bertscore_f1_avg,
    }


def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rougeLsum"]

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        # init RougeScorer once (https://github.com/EleutherAI/lm-evaluation-harness/issues/1692)--rouge_types are constant
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER

    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


def bertscore_f1(references, predictions):
    """Computes the F1 score of the BERTScore metric.
    Args:
        references: A list of reference strings.
        predictions: A list of predicted strings.
        **kwargs: Additional keyword arguments.
    Returns:
        The F1 score of the BERTScore metric.
    """
    global BERTSCORE
    if BERTSCORE is None:
        # init BERTScore once
        BERTSCORE = load("bertscore")
    bertscore = BERTSCORE
    return bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="bert-base-multilingual-cased",
        num_layers=9,
    )["f1"][0]
