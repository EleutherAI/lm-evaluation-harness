"""
The Flores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation
https://aclanthology.org/2022.tacl-1.30/

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, 
Marc’Aurelio Ranzato, Francisco Guzmán, and Angela Fan. 2022. The Flores-101 Evaluation Benchmark for 
Low-Resource and Multilingual Machine Translation. Transactions of the Association for Computational Linguistics, 
10:522–538.

FLORES-101 is a Many-to-Many multilingual translation benchmark dataset for 101 languages.

Github: https://github.com/facebookresearch/flores
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@article{goyal-etal-2022-flores,
    title = "The {F}lores-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation",
    author = "Goyal, Naman  and
      Gao, Cynthia  and
      Chaudhary, Vishrav  and
      Chen, Peng-Jen  and
      Wenzek, Guillaume  and
      Ju, Da  and
      Krishnan, Sanjana  and
      Ranzato, Marc{'}Aurelio  and
      Guzm{\'a}n, Francisco  and
      Fan, Angela",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "10",
    year = "2022",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/2022.tacl-1.30",
    doi = "10.1162/tacl_a_00474",
    pages = "522--538",
}}
"""


class Flores_101_mt(PromptSourceTask):

    DATASET_PATH = "gsarti/flores_101"
    DATASET_NAME = "all"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["dev"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["devtest"]

    def max_generation_length(self):
        return 512
