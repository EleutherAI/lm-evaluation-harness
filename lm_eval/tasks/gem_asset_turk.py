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
import numpy as np
from lm_eval import metrics, utils
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

    def max_generation_length(self):
        return 200

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        answer_choices_list = self.prompt.get_answer_choices_list(doc)
        target = self.doc_to_target(doc)
        if answer_choices_list:
            # If answer_choices_list, then this is a ranked choice prompt.
            # NOTE: In the future, target will be a list of strings.
            # For now, we can assume there will be only 1 target, but its possible
            # that this not the case so we should check for that.
            assert isinstance(target, list) and len(target) == 1
            target = target[0].strip()

            pred = answer_choices_list[np.argmax(results)]
            out = {}

            for metric in self.prompt.metadata.metrics:
                assert (
                    metric in self.CONFIGURED_RANKED_CHOICE_PS_METRICS
                ), "Unexpected metric. Add it, or use a task-specific solution."
                if metric == "Accuracy":
                    out["acc"] = pred == target
            # TODO: Add metrics here.
        else:
            # If not, then this is a generation prompt.
            # NOTE: In the future, target will be a list of strings.
            assert isinstance(target, list)
            pred = results[0].strip()
            out = {}
            self.prompt.metadata.metrics.append("SARI")
            for metric in self.prompt.metadata.metrics:
                if metric == "BLEU":
                    out["bleu"] = (target, pred)
                elif metric == "ROUGE":
                    # TODO: This computes all rouge sub-metrics. Find a generic
                    # way to handle user specified rouge sub-metrics to avoid extra
                    # compute.
                    rouge_scores = metrics.rouge(target, pred)
                    # Flatten rouge score dict.
                    rouge_scores = utils.flatten(rouge_scores)
                    # Merge all the rouge-type scores into the `out` dict.
                    out = {**out, **rouge_scores}
                elif metric == "SARI":
                    out["sari"] = metrics.compute_sari(
                        sentence_to_simplifiy=doc["source"],
                        generated_sentence=pred,
                        references=target,
                    ):
        # TODO: Wrap process results s.t. override impl do not
        # override the save examples.
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
                "answer_choices_list": answer_choices_list,
            }
            return out, example
        return out


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
