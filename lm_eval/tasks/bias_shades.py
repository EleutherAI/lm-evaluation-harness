"""
Multilingual dataset for measuring social biases in language models.
https://huggingface.co/datasets/BigScienceBiasEval/bias-shades/viewer/spanish/test

TODO: Add `arabic`, `german`, `russian`, and `tamil` subsets when `promptsource`
templates become available.
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """"""


class BiasShadesBase(PromptSourceTask):
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


class BiasShadesEnglish(BiasShadesBase):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "english"


class BiasShadesFrench(BiasShadesBase):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "french"


class BiasShadesHindi(BiasShadesBase):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "hindi"


class BiasShadesSpanish(BiasShadesBase):
    VERSION = 0
    DATASET_PATH = "BigScienceBiasEval/bias-shades"
    DATASET_NAME = "spanish"


BIAS_SHADES_CLASSES = [
    BiasShadesEnglish,
    BiasShadesFrench,
    BiasShadesHindi,
    BiasShadesSpanish,
]


def construct_tasks():
    tasks = {}
    for bias_shades_class in BIAS_SHADES_CLASSES:
        tasks[f"bias_shades_{bias_shades_class.DATASET_NAME}"] = bias_shades_class
    return tasks
