"""
A dataset of Legal Documents from German federal court decisions for Named Entity Recognition.
https://arxiv.org/abs/2003.13016v1

The dataset is human-annotated with 19 fine-grained entity classes.
The dataset consists of approx. 67,000 sentences and contains 54,000 annotated entities.
NER tags use the BIO tagging scheme.

Dataset: https://huggingface.co/datasets/elenanereiss/german-ler

NOTE: This dataset is used as language modeling tasks (perplexity) and NOT named entity recogniton.

"""
import re
from lm_eval.base import PerplexityTask


_CITATION = """
@misc{german-ler,
  doi = {10.48550/ARXIV.2003.13016},
  url = {https://arxiv.org/abs/2003.13016},
  author = {Leitner, Elena and Rehm, Georg and Moreno-Schneider, Julián},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {A Dataset of German Legal Documents for Named Entity Recognition},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
"""


class GermanLERPerplexity(PerplexityTask):
    VERSION = 0
    DATASET_PATH = "elenanereiss/german-ler"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return " ".join(doc["tokens"])

    def doc_to_target(self, doc):
        return doc

    def should_decontaminate(self):
        return True

    def count_words(self, doc):
        # count number of words in doc
        return len(re.split(r"\s+", doc))
