import string
from collections import Counter
import re
import pdb
from sklearn.metrics import f1_score


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(il|lo|la|i|gli|le|un|uno|una|del|dello|della|dei|degli|delle)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _f1_score(prediction, ground_truth):
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def squad_em(predictions, references):
    valid_targets = references[0].split(" ||| ")
    exact_matches = [
        1 if normalize_text(predictions[0]) == normalize_text(vt) else 0
        for vt in valid_targets
    ]
    return max(exact_matches)

def squad_f1(predictions, references):
    valid_targets = references[0].split(" ||| ")
    scores = [
        _f1_score(predictions[0], vt)
        for vt in valid_targets
    ]
    return max(scores)

def _rouge(reference, hypothesis, variant):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer([variant], use_stemmer=False)
    score = scorer.score(reference, hypothesis)
    return score[variant].fmeasure

def rouge1(predictions, references):
    return _rouge(references[0], predictions[0], "rouge1")

def rouge2(predictions, references):
    return _rouge(references[0], predictions[0], "rouge2")

def rougeL(predictions, references):
    return _rouge(references[0], predictions[0], "rougeL")


from bert_score import BERTScorer
scorer = BERTScorer(
    model_type="dbmdz/bert-base-italian-xxl-uncased",
    lang="it",
    num_layers=10,
    baseline_path="./lm_eval/tasks/ita_eval/bertscore_baseline_ita.tsv",
    use_fast_tokenizer=True,
    rescale_with_baseline=True,
)

def bertscore(predictions, references):
    return scorer.score(
        predictions,
        references,
        batch_size=16,
    )[-1].item()


def macro_f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="macro")
    return fscore
