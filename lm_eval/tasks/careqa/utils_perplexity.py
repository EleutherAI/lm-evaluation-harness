import re
import math

def doc_to_target(doc) -> str:
    return doc["answer"]


def process_results(doc, results):
    (loglikelihood,) = results
    _words = len(re.split(r"\s+", doc_to_target(doc)))
    _bytes = len(doc_to_target(doc).encode("utf-8"))
    print(f'perplexity: {math.exp(-loglikelihood / _words)}')
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }


