def process_multilabel(doc, results):
    # results contains log-likelihoods for each choice
    # Find the index with highest likelihood
    predicted_idx = results.index(max(results))
    correct_indices = doc["correct_indices"]
    
    # Check if prediction is among correct answers
    is_correct = predicted_idx in correct_indices
    
    return {"acc": int(is_correct)}