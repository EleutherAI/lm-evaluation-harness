import re

from lm_eval.tasks.mts_dialog.utils import doc_to_target


def process_results(doc, results):
    (loglikelihood,) = results
    _words = len(re.split(r"\s+", doc_to_target(doc)))
    _bytes = len(doc_to_target(doc).encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
