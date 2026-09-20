"""OEQ aggregators for the ArabicCulturalQA lm-eval-harness tasks.

Each aggregator is called once at the end of evaluation with the full list of
items returned by `utils.process_results_oeq`. Items are `(gold, pred, lang)`
triples; `lang` is `"ar"` for any Arabic dialect (msa, egyptian, gulf,
levantine, maghrebi) and `"en"` for English.

- BERTScore: `aubmindlab/bert-base-arabertv2` (12 layers) for Arabic,
  `bert-base-uncased` for English. Mirrors the eval setup in the paper.
- ROUGE-L: same arabertv2 tokenizer for Arabic so it aligns with BERTScore
  tokenization; default tokenizer for English.

Heavy text normalization from the project's `qa_eval.py` (camel-tools,
NLTK stem/lemmatize) is intentionally left out so the harness numbers are
reproducible from upstream packages alone. Users who want the paper's full
normalization can post-process the predictions with `qa_eval.py`.
"""

from functools import lru_cache
from types import SimpleNamespace


# ---- BERTScore -------------------------------------------------------------


def _bertscore_model(lang):
    return (
        ("aubmindlab/bert-base-arabertv2", 12)
        if lang == "ar"
        else ("bert-base-uncased", None)
    )


def bertscore_f1(items):
    """Mean BERTScore F1 over the corpus."""
    from bert_score import score as _score

    refs, hyps, langs = zip(*items, strict=True)
    lang = langs[0]
    model, num_layers = _bertscore_model(lang)
    kwargs = {"cands": list(hyps), "refs": list(refs), "model_type": model}
    if num_layers:
        kwargs["num_layers"] = num_layers
    _, _, F = _score(**kwargs)
    return float(F.mean())


# ---- ROUGE-L ---------------------------------------------------------------


@lru_cache(maxsize=1)
def _arabertv2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")


def _rouge_scorer(lang):
    from rouge_score import rouge_scorer

    tok_obj = None
    if lang == "ar":
        try:
            t = _arabertv2_tokenizer()
            tok_obj = SimpleNamespace(tokenize=t.tokenize)
        except (OSError, ImportError):
            # Arabertv2 tokenizer download or load failed; fall back to default.
            tok_obj = None
    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=tok_obj)


def rougeL_f1(items):
    """Mean ROUGE-L F1 over the corpus."""
    refs, hyps, langs = zip(*items, strict=True)
    lang = langs[0]
    scorer = _rouge_scorer(lang)
    scores = [
        scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps, strict=True)
    ]
    return float(sum(scores) / max(1, len(scores)))
