from lm_eval.base import PromptSourceTask
_CITATION = """
@inproceedings{petroni2019language, title={Language Models as Knowledge Bases?},
               author={F. Petroni, T. Rockt{"{a}}schel, A. H. Miller, P. Lewis, A. Bakhtin, Y. Wu and S. Riedel},
               booktitle={In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2019}, year={2019} }

@inproceedings{petroni2020how,
               title={How Context Affects Language Models' Factual Predictions},
               author={Fabio Petroni and Patrick Lewis and Aleksandra Piktus and Tim Rockt{"a}schel and Yuxiang Wu and Alexander H. Miller and Sebastian Riedel},
               booktitle={Automated Knowledge Base Construction}, year={2020}, url={https://openreview.net/forum?id=025X0zPfn} }
"""


class BigScienceLAMA(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "janck/bigscience-lama"
    DATASET_NAME = None


    def has_training_docs(self):
        # TODO: Fill in the return with `True` if the Task has training data; else `False`.
        return False
    def has_validation_docs(self):
        # TODO: Fill in the return with `True` if the Task has validation data; else `False`.
        return False
    def has_test_docs(self):
        # TODO: Fill in the return with `True` if the Task has test data; else `False`.
        return True
    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["train"]

    def test_docs(self):
        if self.has_test_docs():
            self._test_docs = list(self.dataset["test"])
            return self._test_docs




