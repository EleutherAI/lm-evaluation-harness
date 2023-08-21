import evaluate

def bleu(references, predictions):
    bleu_fn = evaluate.load('bleu')
    return bleu_fn.compute(predictions=predictions, references=references)
