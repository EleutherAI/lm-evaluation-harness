import re

from lm_eval.tasks.mimic_iii.mimic_repsum.utils import doc_to_target_clean


def process_results(doc, results):
    (loglikelihood,) = results
    _words = len(re.split(r"\s+", doc_to_target_clean(doc)))
    _bytes = len(doc_to_target_clean(doc).encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
