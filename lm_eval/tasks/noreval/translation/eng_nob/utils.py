def process_results(doc, results):
    # results[0] holds the K sampled translations; pair each with the reference
    pairs = [(doc["targetString"], prediction) for prediction in results[0]]
    return {"bleu": pairs, "chrf": pairs, "bleu_stderr": pairs, "chrf_stderr": pairs}
