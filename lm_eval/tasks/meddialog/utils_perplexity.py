import re

from lm_eval.tasks.meddialog.utils import doc_to_target_qsumm, doc_to_target_raw


def process_results_qsumm(doc, results):
    (loglikelihood,) = results
    _words = len(re.split(r"\s+", doc_to_target_qsumm(doc)))
    _bytes = len(doc_to_target_qsumm(doc).encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }


def process_results_raw(doc, results):
    (loglikelihood,) = results
    _words = len(re.split(r"\s+", doc_to_target_raw(doc)))
    _bytes = len(doc_to_target_raw(doc).encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
