"""
Scitail: A textual entailment dataset from science question answering
https://arxiv.org/pdf/1910.14599.pdf

The SciTail dataset is an entailment dataset created from multiple-choice science exams and web sentences. Each question and the correct answer choice are converted into an assertive statement to form the hypothesis.

Homepage: "https://allenai.org/data/scitail"
"""
import numpy as np
from lm_eval.base import rf, PromptSourceTask
from lm_eval.metrics import mean


_CITATION = """

@inproceedings{khot2018scitail,
  title={Scitail: A textual entailment dataset from science question answering},
  author={Khot, Tushar and Sabharwal, Ashish and Clark, Peter},
  booktitle={Thirty-Second AAAI Conference on Artificial Intelligence},
  year={2018}
}
"""

class SciTailBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "scitail"
    DATASET_NAME = None
    SPLIT = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset[self.SPLIT]["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset[self.SPLIT]["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset[self.SPLIT]["test"]


class SciTailSNLI(SciTailBase):
    SPLIT = "snli_format"


class SciTailTSV(SciTailBase):
    SPLIT = "tsv_format"


class SciTailDGEM(SciTailBase):
    SPLIT = "dgem_format"

class SciTailPredictor(SciTailBase):
    SPLIT = "predictor_format"
