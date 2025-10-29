import sacrebleu

def postprocess_generation(text):
    """Clean model output before scoring"""
    text = text.strip()
    return text.replace("</s>", "").strip()

def metric_bleu(predictions, references):
    """Calculate BLEU score using sacrebleu"""
    references_list = [[ref] for ref in references]
    bleu_score = sacrebleu.corpus_bleu(predictions, references_list)
    return {"bleu": bleu_score.score}

def metric_chrf(predictions, references):
    """Calculate CHRF score using sacrebleu"""
    references_list = [[ref] for ref in references]
    chrf_score = sacrebleu.corpus_chrf(predictions, references_list)
    return {"chrf": chrf_score.score}