"""
What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams
https://arxiv.org/abs/2009.13081

This contains the English portion of the full MedQA dataset, containing 12,723 multiple (4) choice questions from the US medical licensing exam.

Homepage: None
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@misc{jin2020disease,
    title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}, 
    author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
    year={2020},
    eprint={2009.13081},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class MedQA_USMLE(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "GBaker/MedQA-USMLE-4-options"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return []

    def validation_docs(self):
        return []

    def test_docs(self):
        return list(map(self._process_doc, self.dataset["train"])) + \
                list(map(self._process_doc, self.dataset["test"]))

    def _process_doc(self, doc):
        return {
            "query": doc["question"],
            "choices": [
                doc["options"]["A"],
                doc["options"]["B"],
                doc["options"]["C"],
                doc["options"]["D"]
                ],
            "gold": ord(doc["answer_idx"])-ord("A"),
        }

    def doc_to_text(self, doc):
        return doc["query"]
