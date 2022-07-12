"""
TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages

TyDi QA is a question answering dataset covering 11 typologically diverse languages with 200K question-answer pairs.

Paper: https://arxiv.org/abs/2003.05002
Homepage: https://ai.google.com/research/tydiqa
"""
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1

from lm_eval.api.task import PromptSourceTask
from lm_eval.api.metric import mean


_CITATION = """
@article{tydiqa,
    title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
    author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and Vitaly Nikolaev and Jennimaria Palomaki}
    year    = {2020},
    journal = {Transactions of the Association for Computational Linguistics}
}
"""


class TyDiQAPrimaryClassification(PromptSourceTask):
    """
    This task uses the primary_task dataset and implements the classification portion of the Minimal Answer Span task.
    Note: Promptsource currently filters out all non-English examples so this task only reports results on English.
    """

    VERSION = 0
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "primary_task"

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

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            text, target = self.prompt_template.apply(doc)
            return False
        except Exception:
            return True


class TyDiQAGoldPGeneration(PromptSourceTask):
    """
    This task uses the Gold Passage (secondary_task) dataset and implements the Gold Passage task described in the paper, in addition to title and question generation tasks.
    Note: Promptsource currently filters out all non-English examples so this task only reports results on English.
    """

    VERSION = 0
    DATASET_PATH = "tydiqa"
    DATASET_NAME = "secondary_task"

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

    def max_generation_length(self):
        return 128

    def compute_score(self, targets, pred, metric=compute_exact, agg=max):
        return agg([metric(t, pred) for t in targets])

    def process_results(self, doc, results):
        targets = self.doc_to_target(doc)
        pred = results[0].strip()
        example = {"target": targets, "pred": pred}
        out = {}

        # Detect cases handled in superclass method
        for metric in self.prompt_template.metadata.metrics:
            if (
                metric
                in self.CONFIGURED_RANKED_CHOICE_PS_METRICS
                | self.CONFIGURED_GENERATION_PS_METRICS
            ):
                if self.save_examples:
                    super_out, super_example = super().process_results(doc, results)
                    example.update(super_example)
                else:
                    super_out = super().process_results(doc, results)
                out.update(super_out)
            elif metric == "Squad":
                # Otherwise implement SQuAD metric computations, based on PIAF's implementation
                agg_exact_match = self.compute_score(targets, pred, compute_exact)
                agg_f1 = self.compute_score(targets, pred, compute_f1)
                out["f1"] = agg_f1
                out["exact_match"] = agg_exact_match

        if self.save_examples:
            return out, example

        return out

    def higher_is_better(self):
        out = {}
        for metric in self.prompt_template.metadata.metrics:
            if (
                metric
                in self.CONFIGURED_RANKED_CHOICE_PS_METRICS
                | self.CONFIGURED_GENERATION_PS_METRICS
            ):
                out.update(super().higher_is_better())
            elif metric == "Squad":
                out["f1"] = True
                out["exact_match"] = True
        return out

    def aggregation(self):
        out = {}
        for metric in self.prompt_template.metadata.metrics:
            if (
                metric
                in self.CONFIGURED_RANKED_CHOICE_PS_METRICS
                | self.CONFIGURED_GENERATION_PS_METRICS
            ):
                out.update(super().aggregation())
            elif metric == "Squad":
                out["f1"] = mean
                out["exact_match"] = mean
        return out

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        # TODO: Remove this when fixed in `promptsource`
        try:
            # Ensure the `apply` returns 2 values.
            text, target = self.prompt_template.apply(doc)
            return False
        except Exception:
            return True
