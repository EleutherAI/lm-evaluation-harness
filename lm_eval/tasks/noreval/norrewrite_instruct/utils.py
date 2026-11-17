from lm_eval.tasks.noreval.noreval_corpus import (  # noqa: F401
    bleu_stderr,
    chrf_stderr,
    mean_bleu,
    mean_chrf,
)


def process_results(doc, results):
    # results[0] holds the K sampled generations; pair each with the reference
    pairs = [(doc["response"], prediction) for prediction in results[0]]
    return {"bleu": pairs, "chrf": pairs, "bleu_stderr": pairs, "chrf_stderr": pairs}
