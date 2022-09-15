"""
Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods
https://arxiv.org/abs/1804.06876

Winograd-schema evaluation of gendered coreference resolution.
The dataset contains pro-stereotypical and anti-stereotypical parts. The difference in accuracy for those two subsets
quatnifies bias.

Homepage: https://uclanlp.github.io/corefBias/overview
"""
import transformers.data.metrics.squad_metrics as squad_metrics

from lm_eval.api.metric import mean
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{zhao-etal-2018-gender,
    title = "Gender Bias in Coreference Resolution: Evaluation and Debiasing Methods",
    author = "Zhao, Jieyu  and
      Wang, Tianlu  and
      Yatskar, Mark  and
      Ordonez, Vicente  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2003",
    doi = "10.18653/v1/N18-2003",
    pages = "15--20",
    abstract = "In this paper, we introduce a new benchmark for co-reference resolution focused on gender bias, WinoBias. Our corpus contains Winograd-schema style sentences with entities corresponding to people referred by their occupation (e.g. the nurse, the doctor, the carpenter). We demonstrate that a rule-based, a feature-rich, and a neural coreference system all link gendered pronouns to pro-stereotypical entities with higher accuracy than anti-stereotypical entities, by an average difference of 21.1 in F1 score. Finally, we demonstrate a data-augmentation approach that, in combination with existing word-embedding debiasing techniques, removes the bias demonstrated by these systems in WinoBias without significantly affecting their performance on existing datasets.",
}
"""


class WinoBias(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "wino_bias"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        pass

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results):
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        target = self.doc_to_target(doc)[0].strip()
        pred = " ".join(results[0].strip().split(" ")[: len(target.split(" "))])

        # The original paper uses F1. In the case of exactly one predicted and one gold mention,
        # F1 and exact match are equivalent.
        em = squad_metrics.compute_exact(target, pred)
        out = {"em": em}

        # TODO: Wrap process results s.t. override impl do not
        # override the save examples.
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
                "answer_choices_list": answer_choices_list,
            }
            return out, example
        return out

    def aggregation(self):
        return {"em": mean}

    def higher_is_better(self):
        return {"em": True}


class WinoBiasType1Pro(WinoBias):
    DATASET_NAME = "type1_pro"


class WinoBiasType1Anti(WinoBias):
    DATASET_NAME = "type1_anti"


class WinoBiasType2Pro(WinoBias):
    DATASET_NAME = "type2_pro"


class WinoBiasType2Anti(WinoBias):
    DATASET_NAME = "type2_anti"
