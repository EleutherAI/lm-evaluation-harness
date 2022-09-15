"""
CoQA: A Conversational Question Answering Challenge
https://arxiv.org/pdf/1808.07042.pdf

CoQA is a large-scale dataset for building Conversational Question Answering
systems. The goal of the CoQA challenge is to measure the ability of machines to
understand a text passage and answer a series of interconnected questions that
appear in a conversation.

Homepage: https://stanfordnlp.github.io/coqa/
"""
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metric import mean
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@misc{reddy2018coqa,
    title={CoQA: A Conversational Question Answering Challenge},
    author={Siva Reddy and Danqi Chen and Christopher D. Manning},
    year={2018},
    eprint={1808.07042},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


class CoQA(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "coqa"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    @staticmethod
    def compute_scores(gold_list, pred):
        # tests for exact match and on the normalised answer (compute_exact)
        # test for overlap (compute_f1)
        f1_sum = 0.0
        em_sum = 0.0
        if len(gold_list) > 1:
            for i in range(len(gold_list)):
                gold_answers = gold_list[0:i] + gold_list[i + 1 :]
                # predictions compared against (n) golds and take maximum
                em_sum += max(
                    squad_metrics.compute_exact(a, pred) for a in gold_answers
                )
                f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_answers)
        else:
            em_sum += max(squad_metrics.compute_exact(a, pred) for a in gold_list)
            f1_sum += max(squad_metrics.compute_f1(a, pred) for a in gold_list)

        return {
            "em": em_sum / max(1, len(gold_list)),
            "f1": f1_sum / max(1, len(gold_list)),
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
