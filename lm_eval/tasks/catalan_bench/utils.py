import re
from itertools import product

import evaluate
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.utils import general_detokenize


def lowercase_first_letter(text):
    return text[0].lower() + text[1:]


def process_doc_nli(dataset):
    def process_fn(doc):
        # Detokenize(remove extra whitespaces)
        doc["premise"] = general_detokenize(doc["premise"]).strip()
        doc["hypothesis"] = general_detokenize(doc["hypothesis"]).strip()
        # Remove last punctuation mark in the premise
        doc["premise"] = (
            doc["premise"][:-1]
            if doc["premise"].endswith((".", ",", "!", "?"))
            else doc["premise"]
        )
        # Lowercase the first letter in the hypothesis
        doc["hypothesis"] = lowercase_first_letter(doc["hypothesis"])
        # Ensure that the hypothesis ends with a dot
        doc["hypothesis"] = (
            (doc["hypothesis"] + ".")
            if not doc["hypothesis"].endswith(".")
            else doc["hypothesis"]
        )
        return doc

    return dataset.map(process_fn)


def process_results_coqcat(doc, results):
    # Get all possible answers and compute the scores
    turn_id = len(doc["questions"])
    answers = [doc["answers"]["input_text"][turn_id - 1]]
    additional_answers_list = doc.get("additional_answers")
    if additional_answers_list:
        for key, additional_answers in additional_answers_list.items():
            if additional_answers["input_text"][turn_id - 1].lower() not in map(
                str.lower, answers
            ):
                answers.append(additional_answers["input_text"][turn_id - 1])

    gold_list = answers
    pred = results[0].strip().split("\n")[0]
    # import code; code.interact(local=dict(globals(), **locals()))

    f1_sum = 0.0
    em_sum = 0.0
    if len(gold_list) > 1:
        for i in range(len(gold_list)):
            gold_answers = gold_list[0:i] + gold_list[i + 1 :]
            # predictions compared against (n) golds and take maximum
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_answers)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
    else:
        em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
        f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)
    # import code; code.interact(local=dict(globals(), **locals()))
    return {
        "em": em_sum / max(1, len(gold_list)),
        "f1": f1_sum / max(1, len(gold_list)),
    }


def process_results_qa(doc, results):
    preds = results[0]
    reference = doc["answers"][0]["text"]
    # import code; code.interact(local=dict(globals(), **locals()))
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}


def process_doc_cabreu(dataset):
    def process_fn(doc):
        # Remove duplicate spaces
        doc["content"] = re.sub(r" +", " ", doc["content"])
        for summary_type, index in product(
            ["abstractive", "extractive", "extreme"], ["a1", "a2", "a3"]
        ):
            doc["summaries"][summary_type][index] = re.sub(
                r" +", " ", doc["summaries"][summary_type][index]
            )
        return doc

    return dataset.map(process_fn)


def process_docs_paraphrases(dataset):
    empty_docs = []

    def _process_doc(doc):
        if doc["sentence1"] not in [None, ""] and doc["sentence2"] not in [None, ""]:
            doc["sentence1"] = general_detokenize(doc["sentence1"]).strip()
            doc["sentence2"] = general_detokenize(doc["sentence2"]).strip()
            # Remove final punctuation mark in the first sentence
            if doc["sentence1"].endswith((".", ",", ";")):
                doc["sentence1"] = doc["sentence1"][:-1]
            # Start the second sentence in lowercase (to be used after "Yes, ...")
            doc["sentence2"] = lowercase_first_letter(doc["sentence2"])
            return doc
        else:
            empty_docs.append(doc)
            return doc

    return dataset.filter(
        lambda doc: doc["sentence1"] not in [None, ""]
        and doc["sentence2"] not in [None, ""]
    ).map(_process_doc)


def process_docs_copa_ca(dataset):
    def _process_doc(doc):
        doc["choice1"] = lowercase_first_letter(doc["choice1"])
        doc["choice2"] = lowercase_first_letter(doc["choice2"])
        return doc

    return dataset.map(_process_doc)


def rouge1(items):
    """
    # passthrough for efficiency
    """
    return items


def rouge1_agg(items):
    """
    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    rouge_scorer = evaluate.load("rouge")
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]
