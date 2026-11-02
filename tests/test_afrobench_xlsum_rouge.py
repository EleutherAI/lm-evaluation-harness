"""ROUGE-L for AfroBench XL-Sum must not drop non-Latin scripts.

rouge_score's default tokenizer keeps only ``[a-z0-9]``, so before the fix an
exact-match Amharic/Arabic/Telugu/Tigrinya summary scored 0.
"""

import importlib

import pytest
from rouge_score import rouge_scorer, scoring
from rouge_score.tokenize import tokenize as rouge_default_tokenize


PROMPT_MODULES = [
    importlib.import_module(f"lm_eval.tasks.afrobench.xlsum.prompt_{p}.utils")
    for p in (1, 2, 3)
]

SENTENCES = {
    "amharic": "የኢትዮጵያ መንግስት አዲስ ፖሊሲ አወጣ",
    "arabic": "أعلنت الحكومة عن سياسة جديدة",
    "telugu": "ప్రభుత్వం కొత్త విధానాన్ని ప్రకటించింది",
    "tigrinya": "መንግስቲ ሓድሽ ፖሊሲ ኣውጺኡ",
    "yoruba": "Ìjọba ti kéde òfin tuntun",
    "swahili": "Serikali imetangaza sera mpya",
}


class _OfflineRouge:
    """Offline stand-in for ``evaluate.load("rouge")``.

    Mirrors the evaluate metric: wraps an optional ``tokenizer`` callable,
    scores with ``rouge_score.RougeScorer`` and returns the bootstrap mid
    F-measure.
    """

    def compute(self, predictions, references, tokenizer=None):
        if tokenizer is not None:
            func = tokenizer

            class _Tok:
                def tokenize(self, text):
                    return func(text)

            tokenizer = _Tok()
        scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=tokenizer)
        aggregator = scoring.BootstrapAggregator()
        for ref, pred in zip(references, predictions, strict=True):
            aggregator.add_scores(scorer.score(ref, pred))
        return {"rougeL": aggregator.aggregate()["rougeL"].mid.fmeasure}


@pytest.fixture(params=PROMPT_MODULES, ids=["prompt_1", "prompt_2", "prompt_3"])
def utils(request, monkeypatch):
    module = request.param
    monkeypatch.setattr(module.evaluate, "load", lambda name, *a, **k: _OfflineRouge())
    return module


@pytest.mark.parametrize("lang", SENTENCES)
def test_exact_match_scores_one(utils, lang):
    text = SENTENCES[lang]
    assert utils.rougeL_agg([(text, text)]) == pytest.approx(1.0)


def test_unrelated_prediction_scores_zero(utils):
    assert utils.rougeL_agg([(SENTENCES["amharic"], "ሰላም ለዓለም")]) == 0.0


def test_default_tokenizer_drops_non_latin_scripts():
    # Documents the root cause: these would score 0 without a custom tokenizer.
    for lang in ("amharic", "arabic", "telugu", "tigrinya"):
        assert rouge_default_tokenize(SENTENCES[lang], None) == []


@pytest.mark.parametrize(
    "text",
    [
        "Serikali imetangaza sera mpya.",
        "The U.S. economy grew 3.5% in Q2_2024 -- (BBC News)",
        "  multiple   spaces\tand\nnewlines  ",
        "",
    ],
)
def test_ascii_tokenization_unchanged(text):
    for module in PROMPT_MODULES:
        assert module._unicode_tokenize(text) == rouge_default_tokenize(text, None)


def test_combining_marks_stay_inside_words():
    # Telugu vowel signs and the virama are combining marks (category M*).
    assert PROMPT_MODULES[0]._unicode_tokenize("కొత్త విధానాన్ని") == [
        "కొత్త",
        "విధానాన్ని",
    ]
