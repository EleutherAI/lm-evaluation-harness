"""
Semantic Noise Matters for Neural Natural Language Generation
http://arxiv.org/abs/1911.03905

A cleaned version of the dataset from the E2E NLG Challenge.
The dataset contains MR with restaurant attributes and corresponding descriptions.

Homepage: https://github.com/tuetschek/e2e-cleaning
"""
from lm_eval.base import PromptSourceTask, rf

_CITATION = """
@inproceedings{dusek-etal-2019-semantic,
    title = "Semantic Noise Matters for Neural Natural Language Generation",
    author = "Du{\v{s}}ek, Ond{\v{r}}ej  and
      Howcroft, David M.  and
      Rieser, Verena",
    booktitle = "Proceedings of the 12th International Conference on Natural Language Generation",
    year = "2019",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W19-8652",
    doi = "10.18653/v1/W19-8652",
    pages = "421--426",
}
"""

# Work in progress
class E2E_NLG_Cleaned(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "e2e_nlg_cleaned"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def stopping_criteria(self):
        return '\n'

    def max_generation_length(self):
        # TODO check
        return 512

    def invalid_doc_for_prompt(self, doc) -> bool:
        """The QA prompts are not applicable to all the examples, we want to filter these out."""
        return self.prompt.name.endswith("_qa")

    def doc_to_text(self, doc) -> str:
        # if the response is not defined in PS, the text will be an empty string
        text = self.prompt.apply(doc)[0]

        return text

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "bleu": metrics.bleu,
            "rouge": metrics.rouge,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "bleu": True,
            "rouge": True,
        }