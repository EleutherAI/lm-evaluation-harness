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
from typing import Optional
from promptsource.templates import Template

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{DBLP:journals/corr/abs-2005-00481,
  author    = {Fernando Alva{-}Manchego and
               Louis Martin and
               Antoine Bordes and
               Carolina Scarton and
               Benoit Sagot and
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


class AssetTurk(PromptSourceTask):

    DATASET_PATH = "GEM/wiki_auto_asset_turk"
    DATASET_NAME = None
    SPLIT = None

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
        prompt_template: Optional[Template] = None,
        example_separator: Optional[str] = "\n###\n",
        text_target_separator: Optional[str] = " ",
        save_examples: Optional[bool] = True,
    ):
        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            prompt_template=prompt_template,
            example_separator=example_separator,
            text_target_separator=text_target_separator,
            save_examples=save_examples,
        )
        # Adding SARI to metrics to list because `promptsource`
        # does not currently support this option.
        if "SARI" not in self.prompt_template.metadata.metrics:
            self.prompt_template.metadata.metrics.append("SARI")

    def doc_to_rawtext(self, doc):
        return doc["source"]

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        return self.dataset[str(self.SPLIT)]

    def max_generation_length(self):
        return 200


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
