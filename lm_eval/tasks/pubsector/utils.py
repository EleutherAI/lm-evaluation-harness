import datasets
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer, scoring


ROUGE_SCORER = None




def process_docs_gen(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(preprocess_function)


def preprocess_function(examples):
    return {
        "question": examples["question"].strip(),
        "context": examples["context"].strip(),
        "answer": examples["answer"].strip(),
    }



def process_results_gen(doc, results):
    completion = results[0].strip()
    ref = doc["answer"]

    # Process the sentence-level BLEURT, BLEU, and ROUGE for similarity measures.

    # # BLEURT
    # bleurt_scores_true = self.bleurt.compute(
    #     predictions=[completion] * len(true_refs), references=true_refs
    # )["scores"]
    # bleurt_scores_false = self.bleurt.compute(
    #     predictions=[completion] * len(false_refs), references=false_refs
    # )["scores"]
    # bleurt_correct = max(bleurt_scores_true)
    # bleurt_incorrect = max(bleurt_scores_false)
    # bleurt_max = bleurt_correct
    # bleurt_diff = bleurt_correct - bleurt_incorrect
    # bleurt_acc = int(bleurt_correct > bleurt_incorrect)

    # BLEU
    bleu_score = bleu([[ref]], [completion]) 
    

    # ROUGE-N
    rouge_scores = rouge([ref], [completion])
    

    return {
        "bleu": bleu_score,
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeLsum"],
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
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"])
    scorer = ROUGE_SCORER

    def _prepare_summary(summary):
        return summary.replace(" . ", ".\n")

    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        aggregator.add_scores(scorer.score(_prepare_summary(ref), _prepare_summary(pred)))
    result = aggregator.aggregate()
    return {t: result[t].mid.fmeasure * 100 for t in ["rouge1", "rouge2", "rougeLsum"]}