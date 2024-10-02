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


def process_results_qa(doc, results):
    preds = results[0]
    reference = doc["answers"]["text"][0]
    # import code; code.interact(local=dict(globals(), **locals()))
    f1_sum = squad_metrics.compute_f1(reference, preds)
    exact_match = squad_metrics.compute_exact(reference, preds)
    return {"f1": f1_sum, "exact_match": exact_match}


def process_xlsum(dataset):
    def _process_doc(doc):
        # Remove double spaces
        doc["text"] = re.sub(r" +", " ", doc["text"])
        doc["summary"] = re.sub(r" +", " ", doc["summary"])
        return doc

    return dataset.map(_process_doc)


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

    if empty_docs != []:
        len_empty_docs = len(empty_docs)
        print(
            f"Found {len_empty_docs} empty documents out of the {len(dataset)} total docs in the dataset: {empty_docs}"
        )
    return dataset.filter(
        lambda doc: doc["sentence1"] not in [None, ""]
        and doc["sentence2"] not in [None, ""]
    ).map(_process_doc)


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
    # import code; code.interact(local=dict(globals(), **locals()))
    return rouge_scorer.compute(predictions=preds, references=refs)["rouge1"]
