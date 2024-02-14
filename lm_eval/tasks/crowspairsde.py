
"""
CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models
https://aclanthology.org/2020.emnlp-main.154/


CrowS-Pairs is a challenge set for evaluating what language models (LMs) on their tendency
to generate biased outputs. CrowS-Pairs comes in 2 languages and the English subset has
a newer version which fixes some issues with the original version.

Homepage: https://github.com/nyu-mll/crows-pairs
"""

from lm_eval.base import rf, Task
from lm_eval.metrics import mean

_CITATION = """
@inproceedings{nangia-etal-2020-crows,
    title = "{C}row{S}-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models",
    author = "Nangia, Nikita  and
      Vania, Clara  and
      Bhalerao, Rasika  and
      Bowman, Samuel R.",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.emnlp-main.154",
    doi = "10.18653/v1/2020.emnlp-main.154",
    pages = "1953--1967",
    abstract = "Pretrained language models, especially masked language models (MLMs) have seen success across many NLP tasks. However, there is ample evidence that they use the cultural biases that are undoubtedly present in the corpora they are trained on, implicitly creating harm with biased representations. To measure some forms of social bias in language models against protected demographic groups in the US, we introduce the Crowdsourced Stereotype Pairs benchmark (CrowS-Pairs). CrowS-Pairs has 1508 examples that cover stereotypes dealing with nine types of bias, like race, religion, and age. In CrowS-Pairs a model is presented with two sentences: one that is more stereotyping and another that is less stereotyping. The data focuses on stereotypes about historically disadvantaged groups and contrasts them with advantaged groups. We find that all three of the widely-used MLMs we evaluate substantially favor sentences that express stereotypes in every category in CrowS-Pairs. As work on building less biased models advances, this dataset can be used as a benchmark to evaluate progress.",
}
"""


class CrowsPairsDE(Task):
    VERSION = 0
    DATASET_PATH = "lamarr-org/crows_pairs_de"
    DATASET_NAME = None
    BIAS_TYPE = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__()
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None
        self.non_unknown_outputs = 0
        self.biased_answers = 0
        self.final_accuracy = None
        self.bias_disambiguate = None
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            test_dataset = self.dataset["test"]
            if self.BIAS_TYPE is not None:
                test_dataset = test_dataset.filter(
                    lambda example: example["bias_type"].startswith(self.BIAS_TYPE)
                )
            return test_dataset

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        assert num_fewshot == num_fewshot
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the  "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
            )

        return ""

    def test_docs(self):
        return self.dataset["test"]

    def _process_doc(self, doc):
        return doc

    def doc_to_text(self, doc):
        return ""

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["sentence_good"] + " " + doc["sentence_bad"]

    def doc_to_target(self, doc):
        return " "

    def construct_requests(self, doc, ctx):
        assert not ctx

        # Calculate the loglikelihood for the more and the less stereotypical sentence.
        # Note that loglikelihood translates the "" prefix to the "<|endoftext|>" token
        return [
            rf.loglikelihood("", doc["sent_more"]),
            rf.loglikelihood("", doc["sent_less"]),
        ]

    def process_results(self, doc, results):
        likelihood1, likelihood2 = results

        # Calculate the absolute difference in loglikelihoods
        diff = abs(likelihood1[0] - likelihood2[0])

        # if stereotypical sentence more likely (loglikelihood higher)
        #  treat this as predicting stereotyped sentence
        acc = 1.0 if likelihood1[0] > likelihood2[0] else 0.0

        return {"likelihood_difference": diff, "pct_stereotype": acc}

    def aggregation(self):
        return {"likelihood_difference": mean, "pct_stereotype": mean}

    def higher_is_better(self):
        # For all metrics lower is better
        return {"likelihood_difference": False, "pct_stereotype": False}
