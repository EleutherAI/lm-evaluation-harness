"""
JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension
https://arxiv.org/abs/2202.01764

Japanese Question Answering Dataset (JaQuAD), released in 2022, is a human-annotated dataset created for Japanese Machine Reading Comprehension.
JaQuAD is developed to provide a SQuAD-like QA dataset in Japanese.
JaQuAD contains 39,696 question-answer pairs.
Questions and answers are manually curated by human annotators.
Contexts are collected from Japanese Wikipedia articles.

Homepage: https://github.com/SkelterLabsInc/JaQuAD
"""
from .jsquad import JSQuAD


_CITATION = """
@misc{so2022jaquad,
      title={{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
      author={ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
      year={2022},
      eprint={2202.01764},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class JaQuAD(JSQuAD):
    VERSION = 0.1
    DATASET_PATH = "SkelterLabsInc/JaQuAD"
    DATASET_NAME = None

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]
    

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        if "answer_type" in doc["answers"]:
            doc["answers"].pop("answer_type")
        return JSQuAD.process_results(self, doc, results)

