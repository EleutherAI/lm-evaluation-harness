import evaluate

def bleu(predictions, references):
    rouge_fn = evaluate.load('bleu')
    results = rouge_fn.compute(predictions=predictions, references=references)
    return results['bleu']
