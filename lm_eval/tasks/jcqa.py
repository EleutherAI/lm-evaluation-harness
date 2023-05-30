"""
JCommonsenseQA, from:
JGLUE: Japanese General Language Understanding Evaluation
https://aclanthology.org/2022.lrec-1.317.pdf

JCommonsenseQA is a Japanese version of CommonsenseQA (Talmor+, 2019), which is a multiple-choice question answering dataset that requires commonsense reasoning ability. It is built using crowdsourcing with seeds extracted from the knowledge base ConceptNet.

Homepage: https://github.com/yahoojapan/JGLUE
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{kurihara-etal-2022-jglue,
    title = "{JGLUE}: {J}apanese General Language Understanding Evaluation",
    author = "Kurihara, Kentaro  and
      Kawahara, Daisuke  and
      Shibata, Tomohide",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    month = jun,
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://aclanthology.org/2022.lrec-1.317",
    pages = "2957--2966",
    abstract = "To develop high-performance natural language understanding (NLU) models, it is necessary to have a benchmark to evaluate and analyze NLU ability from various perspectives. While the English NLU benchmark, GLUE, has been the forerunner, benchmarks are now being released for languages other than English, such as CLUE for Chinese and FLUE for French; but there is no such benchmark for Japanese. We build a Japanese NLU benchmark, JGLUE, from scratch without translation to measure the general NLU ability in Japanese. We hope that JGLUE will facilitate NLU research in Japanese.",
}
"""

class JCommonsenseQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "shunk031/JGLUE" 
    DATASET_NAME = "JCommonsenseQA"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])


    def get_keys(self):
        return ["A", "B", "C", "D", "E"]

    def _process_doc(self, doc):
        def format_example(doc):
            """
            """
            question = doc["question"]
            choices = "".join(
                [f"{key}. {choice}\n" for key, choice in zip(self.get_keys(), [doc['choice0'],doc['choice1'],doc['choice2'],doc['choice3'],doc['choice4']])]
            )
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        return {
            "query": format_example(doc),  # The query prompt.
            "choices": self.get_keys(),  # The list of choices.
            "gold": doc['label'], 
        }

    def doc_to_text(self, doc):
        return doc["query"]
    
    def fewshot_context(self, doc, num_fewshot, **kwargs):
        description = f"The following are multiple choice questions (with answers)."

        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)