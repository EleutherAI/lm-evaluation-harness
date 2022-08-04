"""
Semantic Noise Matters for Neural Natural Language Generation
http://arxiv.org/abs/1911.03905

A cleaned version of the dataset from the E2E NLG Challenge.
The dataset contains MR with restaurant attributes and corresponding descriptions.

Homepage: https://github.com/tuetschek/e2e-cleaning
"""
from lm_eval.api.task import PromptSourceTask


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
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 64

    def invalid_doc_for_prompt(self, doc) -> bool:
        """The QA prompts are not applicable to all the examples, we want to filter these out."""
        # HACK: Some templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            text, target = self.prompt_template.apply(doc)
            return (
                self.prompt_template.name.endswith("_qa")
                or self.prompt_template.name == "family_friendly_yes_no"
            )
        except Exception:
            return True

    def doc_to_text(self, doc) -> str:
        # if the response is not defined in PS, the text will be a single-element list containing an empty string
        text = self.prompt_template.apply(doc)[0]
        return text
