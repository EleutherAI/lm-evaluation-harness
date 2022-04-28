"""
ASSET: ASSET (Alva-Manchego et al., 2020) is multi-reference dataset
for the evaluation of sentence simplification in English. The dataset
uses the same 2,359 sentences from TurkCorpus (Xu et al., 2016)
and each sentence is associated with 10 crowdsourced simplifications.
Unlike previous simplification datasets, which contain a single
transformation (e.g., lexical paraphrasing in TurkCorpus or sentence
splitting in HSplit), the simplifications in ASSET encompass a variety
of rewriting transformations.
https://aclanthology.org/2020.acl-main.424.pdf

TurkCorpus: TURKCorpus is a multi-reference dataset for the evaluation of
sentence simplification in English. The dataset consists of 2,359 sentences
from the Parallel Wikipedia Simplification (PWKP) corpus. Each sentence is
associated with 8 crowdsourced simplifications that focus on only lexical
paraphrasing (no sentence splitting or deletion).
https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf
"""
from lm_eval.base import PromptSourceTask

_CITATION = """
@article{DBLP:journals/corr/abs-2005-00481,
  author    = {Fernando Alva{-}Manchego and
               Louis Martin and
               Antoine Bordes and
               Carolina Scarton and
               Beno{\^{\i}}t Sagot and
               Lucia Specia},
  title     = {{ASSET:} {A} Dataset for Tuning and Evaluation of Sentence Simplification
               Models with Multiple Rewriting Transformations},
  journal   = {CoRR},
  volume    = {abs/2005.00481},
  year      = {2020},
  url       = {https://arxiv.org/abs/2005.00481},
  eprinttype = {arXiv},
  eprint    = {2005.00481},
  timestamp = {Thu, 14 Oct 2021 16:38:25 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2005-00481.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}"""

""""@article{Xu-EtAl:2016:TACL,
 author = {Wei Xu and Courtney Napoles and Ellie Pavlick and Quanze Chen and Chris Callison-Burch},
 title = {Optimizing Statistical Machine Translation for Text Simplification},
 journal = {Transactions of the Association for Computational Linguistics},
 volume = {4},
 year = {2016},
 url = {https://cocoxu.github.io/publications/tacl2016-smt-simplification.pdf},
 pages = {401--415}
 }"""


class AssetTurk(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/wiki_auto_asset_turk"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        return self.dataset[str(self.SPLIT)]

    # def stopping_criteria(self):
    #     return None

    def max_generation_length(self):
        return 200

    # def higher_is_better(self):
    #     return {"bleu": True, "rouge": True}


class AssetTest(AssetTurk):
    SPLIT = "test_asset"


class TurkTest(AssetTurk):
    SPLIT = "test_turk"


class AssetTest1(AssetTurk):
    SPLIT = "challenge_test_asset_backtranslation"


class AssetTest2(AssetTurk):
    SPLIT = "challenge_test_asset_bfp02"


class AssetTest3(AssetTurk):
    SPLIT = "challenge_test_asset_bfp05"


class AssetTest4(AssetTurk):
    SPLIT = "challenge_test_asset_nopunc"


class TurkTest1(AssetTurk):
    SPLIT = "challenge_test_turk_backtranslation"


class TurkTest2(AssetTurk):
    SPLIT = "challenge_test_turk_bfp02"


class TurkTest3(AssetTurk):
    SPLIT = "challenge_test_turk_bfp05"


class TurkTest4(AssetTurk):
    SPLIT = "challenge_test_turk_nopunc"


ASSET_TURK_CLASSES = [
    AssetTest,
    TurkTest,
    TurkTest1,
    TurkTest2,
    TurkTest3,
    TurkTest4,
    AssetTest1,
    AssetTest2,
    AssetTest3,
    AssetTest4,
]


def construct_tasks():
    tasks = {}
    for asset_turk_class in ASSET_TURK_CLASSES:
        tasks[f"GEM/wiki_auto_asset_turk_{asset_turk_class.SPLIT}"] = asset_turk_class
    return tasks
