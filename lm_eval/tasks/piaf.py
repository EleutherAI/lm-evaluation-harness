"""
Project PIAF: Building a Native French Question-Answering Dataset
https://arxiv.org/pdf/2007.00968.pdf

Piaf is a reading comprehension dataset. This version, published in February 2020,
contains 3835 questions on French Wikipedia.

Homepage: https://huggingface.co/datasets/piaf
"""
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.task import PromptSourceTask
from lm_eval.api.metric import mean


_CITATION = """
@InProceedings{keraron-EtAl:2020:LREC,
  author    = {Keraron, Rachel  and  Lancrenon, Guillaume  and  Bras, Mathilde  and  Allary, FrÃ©dÃ©ric  and  Moyse, Gilles  and  Scialom, Thomas  and  Soriano-Morales, Edmundo-Pavel  and  Staiano, Jacopo},
  title     = {Project PIAF: Building a Native French Question-Answering Dataset},
  booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2020},
  address        = {Marseille, France},
  publisher      = {European Language Resources Association},
  pages     = {5483--5492},
  abstract  = {Motivated by the lack of data for non-English languages, in particular for the evaluation of downstream tasks such as Question Answering, we present a participatory effort to collect a native French Question Answering Dataset. Furthermore, we describe and publicly release the annotation tool developed for our collection effort, along with the data obtained and preliminary baselines.},
  url       = {https://www.aclweb.org/anthology/2020.lrec-1.673}
}
"""


class PIAF(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "piaf"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

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
            return self.dataset["train"]

    def max_generation_length(self):
        return 128

    @staticmethod
    def compute_scores(doc, pred):
        # tests for exact match and on the normalised answer (compute_exact)
        # test for overlap (compute_f1)

        em_sum = squad_metrics.compute_exact(doc[0], pred)
        f1_sum = squad_metrics.compute_f1(doc[0], pred)

        return {
            "em": em_sum,
            "f1": f1_sum,
        }

    def process_results(self, doc, results):
        targets = self.doc_to_target(doc)
        pred = results[0].strip().split("\n")[0]
        scores = self.compute_scores(targets, pred)

        out = {
            "f1": scores["f1"],
            "em": scores["em"],
        }

        if self.save_examples:
            example = {"target": targets, "pred": pred}
            return out, example
        return out

    def higher_is_better(self):
        return {
            "f1": True,
            "em": True,
        }

    def aggregation(self):
        return {
            "f1": mean,
            "em": mean,
        }
