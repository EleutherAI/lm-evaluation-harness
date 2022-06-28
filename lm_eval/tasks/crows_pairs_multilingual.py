"""
French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English
https://hal.inria.fr/hal-03629677/file/ACLFinal.pdf
Measuring social biases in masked language models in English and French.
https://gitlab.inria.fr/french-crows-pairs/acl-2022-paper-data-and-code/-/tree/main
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """\
@inproceedings{neveol2022french,
  title={French CrowS-Pairs: Extending a challenge dataset for measuring social bias in masked language models to a language other than English},
  author={N{\'e}v{\'e}ol, Aur{\'e}lie and Dupont, Yoann and Bezancon, Julien and Fort, Kar{\"e}n},
  booktitle={ACL 2022-60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
"""


class CrowsPairsEnglish(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/crows_pairs_multilingual"
    DATASET_NAME = "english"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]


class CrowsPairsFrench(PromptSourceTask):

    DATASET_PATH = "BigScienceBiasEval/crows_pairs_multilingual"
    DATASET_NAME = "french"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        pass

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
