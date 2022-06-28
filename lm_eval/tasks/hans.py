"""
Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference
https://arxiv.org/abs/1902.01007

A controlled evaluation set called HANS (Heuristic Analysis for NLI Systems),
which contains many examples where the heuristics fail.

Homepage: https://github.com/tommccoy1/hans
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{mccoy-etal-2019-right,
    title = "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference",
    author = "McCoy, Tom  and
      Pavlick, Ellie  and
      Linzen, Tal",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1334",
    doi = "10.18653/v1/P19-1334",
    pages = "3428--3448",
    abstract = "A machine learning system can score well on a given test set by relying on heuristics that are effective for frequent example types but break down in more challenging cases. We study this issue within natural language inference (NLI), the task of determining whether one sentence entails another. We hypothesize that statistical NLI models may adopt three fallible syntactic heuristics: the lexical overlap heuristic, the subsequence heuristic, and the constituent heuristic. To determine whether models have adopted these heuristics, we introduce a controlled evaluation set called HANS (Heuristic Analysis for NLI Systems), which contains many examples where the heuristics fail. We find that models trained on MNLI, including BERT, a state-of-the-art model, perform very poorly on HANS, suggesting that they have indeed adopted these heuristics. We conclude that there is substantial room for improvement in NLI systems, and that the HANS dataset can motivate and measure progress in this area.",
}
"""


class HANS(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "hans"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def process_results(self, doc, results):
        out, log = super().process_results(doc, results)
        log["subcase"] = doc["subcase"]
        log["heuristic"] = doc["heuristic"]
        return out, log
