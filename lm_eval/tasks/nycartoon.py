"""
The New Yorker Caption Contest
https://arxiv.org/pdf/2209.06293.pdf

Researchers challenge AI models to "demonstrate understanding" of the humor in
the New Yorker's regular cartoon captioning contest. In this sub-task for
text-based models, the model should match a human-written description to the
associated caption from five options.

Homepage: https://www.capcon.dev/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{hessel2022androids,
  title={Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest},
  author={Hessel, Jack and Marasovi{\'c}, Ana and Hwang, Jena D and Lee, Lillian and Da, Jeff and Zellers, Rowan and Mankoff, Robert and Choi, Yejin},
  journal={arXiv preprint arXiv:2209.06293},
  year={2022}
}
"""


class NYCaption(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "jmhessel/newyorker_caption_contest"
    DATASET_NAME = "matching"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": "Cartoon description: " + doc["image_description"] + "\nCaption:",
            "choices": doc["caption_choices"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["label"]),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]
