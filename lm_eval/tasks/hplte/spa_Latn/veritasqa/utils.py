import datasets
import numpy as np


try:
    import sacrebleu
    from rouge_score import rouge_scorer, scoring
except ModuleNotFoundError as e:
    raise type(e)(
        "`sacrebleu` and `rouge_score` are required for evaluating the model on NorEval."
    ) from e


ROUGE_SCORER = None


LETTERS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
]


def mc1_p1(doc):
    choices = doc["mc1_targets"]["choices"]
    formatted_choices = "\n".join(
        [f"Opción {LETTERS[i]}: {choice}" for i, choice in enumerate(choices)]
    )
    letters = LETTERS[: len(choices)]
    return f"Pregunta: {doc['question']}\n{formatted_choices}\n¿Es la respuesta correcta {', '.join(letters[:-1])} o {letters[-1]}?\nRespuesta:"


def mc2_p1(doc):
    choices = doc["mc2_targets"]["choices"]
    formatted_choices = "\n".join(
        [f"Opción {LETTERS[i]}: {choice}" for i, choice in enumerate(choices)]
    )
    letters = LETTERS[: len(choices)]
    return f"Pregunta: {doc['question']}\n{formatted_choices}\n¿Es la respuesta correcta {', '.join(letters[:-1])} o {letters[-1]}?\nRespuesta:"


def mc1_p2(doc):
    choices = doc["mc1_targets"]["choices"]
    formatted_choices = "".join([f"\n- {choice}" for choice in choices])
    return f"Pregunta: {doc['question']}\nElige la respuesta correcta de la lista:\n{formatted_choices}\n¿Cuál es la respuesta correcta?\nRespuesta:"


def mc2_p2(doc):
    choices = doc["mc2_targets"]["choices"]
    formatted_choices = "".join([f"\n- {choice}" for choice in choices])
    return f"Pregunta: {doc['question']}\nElige la respuesta correcta de la lista:\n{formatted_choices}\n¿Cuál es la respuesta correcta?\nRespuesta:"


def doc_to_choice_mc1_p1(doc):
    choices = doc["mc1_targets"]["choices"]
    return [f"Opción {letter}" for letter in LETTERS[: len(choices)]]


def doc_to_choice_mc2_p1(doc):
    choices = doc["mc2_targets"]["choices"]
    return [f"Opción {letter}" for letter in LETTERS[: len(choices)]]


def process_results_mc2(doc, results):
    lls, _ = zip(*results, strict=False)
    # Split on the first `0` as everything before it is true (`1`).
    split_idx = list(doc["mc2_targets"]["labels"]).index(0)
    # Compute the normalized probability mass for the correct answer.
    ll_true, ll_false = lls[:split_idx], lls[split_idx:]
    p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
    p_true = p_true / (sum(p_true) + sum(p_false))
    return {"acc": sum(p_true)}


def format_answers(answers):
    formatted_answers = []
    for answer in answers:
        answer = answer.strip()
        if len(answer):
            # Add a period after all answers.
            if answer[-1] != ".":
                formatted_answers.append(answer + ".")
            else:
                formatted_answers.append(answer)
    return formatted_answers


def process_docs_veritasqa(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(preprocess_function_veritasqa)


def preprocess_function_veritasqa(doc):
    doc["question"] = doc["question"].strip()
    correct_answers = format_answers(
        [x.strip() for x in doc["correct_answers"].split(";")]
    )
    incorrect_answers = format_answers(
        [x.strip() for x in doc["incorrect_answers"].split(";")]
    )
    mc1_targets = {
        "choices": [correct_answers[0]] + incorrect_answers,
        "labels": [1] + [0] * (len(incorrect_answers)),
    }
    mc2_targets = {
        "choices": correct_answers + incorrect_answers,
        "labels": [1] * len(correct_answers) + [0] * len(incorrect_answers),
    }
    doc["mc1_targets"] = mc1_targets
    doc["mc2_targets"] = mc2_targets
    doc["correct_answers"] = correct_answers
    doc["incorrect_answers"] = incorrect_answers
    return doc


def process_results_gen(doc, results):
    completion = results[0]
    true_refs, false_refs = doc["correct_answers"], doc["incorrect_answers"]
    all_refs = true_refs + false_refs
    # BLEU
    bleu_scores = [bleu([[ref]], [completion]) for ref in all_refs]
    bleu_correct = np.nanmax(bleu_scores[: len(true_refs)])
    bleu_incorrect = np.nanmax(bleu_scores[len(true_refs) :])
    bleu_max = bleu_correct
    bleu_diff = bleu_correct - bleu_incorrect
    bleu_acc = int(bleu_correct > bleu_incorrect)
    # ROUGE-N
    rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
    # ROUGE-1
    rouge1_scores = [score["rouge1"] for score in rouge_scores]
    rouge1_correct = np.nanmax(rouge1_scores[: len(true_refs)])
    rouge1_incorrect = np.nanmax(rouge1_scores[len(true_refs) :])
    rouge1_max = rouge1_correct
    rouge1_diff = rouge1_correct - rouge1_incorrect
    rouge1_acc = int(rouge1_correct > rouge1_incorrect)
    # ROUGE-2
    rouge2_scores = [score["rouge2"] for score in rouge_scores]
    rouge2_correct = np.nanmax(rouge2_scores[: len(true_refs)])
    rouge2_incorrect = np.nanmax(rouge2_scores[len(true_refs) :])
    rouge2_max = rouge2_correct
    rouge2_diff = rouge2_correct - rouge2_incorrect
    rouge2_acc = int(rouge2_correct > rouge2_incorrect)
    # ROUGE-L
    rougeL_scores = [score["rougeLsum"] for score in rouge_scores]
    rougeL_correct = np.nanmax(rougeL_scores[: len(true_refs)])
    rougeL_incorrect = np.nanmax(rougeL_scores[len(true_refs) :])
    rougeL_max = rougeL_correct
    rougeL_diff = rougeL_correct - rougeL_incorrect
    rougeL_acc = int(rougeL_correct > rougeL_incorrect)
    return {
        "bleu_max": bleu_max,
        "bleu_acc": bleu_acc,
        "bleu_diff": bleu_diff,
        "rouge1_max": rouge1_max,
        "rouge1_acc": rouge1_acc,
        "rouge1_diff": rouge1_diff,
        "rouge2_max": rouge2_max,
        "rouge2_acc": rouge2_acc,
        "rouge2_diff": rouge2_diff,
        "rougeL_max": rougeL_max,
        "rougeL_acc": rougeL_acc,
        "rougeL_diff": rougeL_diff,
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
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    # Add newlines between sentences to correctly compute `rougeLsum`.

    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        # init RougeScorer once (https://github.com/EleutherAI/lm-evaluation-harness/issues/1692)--rouge_types are constant
        ROUGE_SCORER = rouge_scorer.RougeScorer(rouge_types)
    scorer = ROUGE_SCORER

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds, strict=False):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}
