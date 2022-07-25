"""
GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding
https://openreview.net/pdf?id=rJ4km2R5t7

The General Language Understanding Evaluation (GLUE) benchmark is a collection of
resources for training, evaluating, and analyzing natural language understanding
systems. GLUE consists of:
- A benchmark of nine sentence- or sentence-pair language understanding tasks built
on established existing datasets and selected to cover a diverse range of dataset
sizes, text genres, and degrees of difficulty, and
- A diagnostic dataset designed to evaluate and analyze model performance with
respect to a wide range of linguistic phenomena found in natural language.

Homepage: https://gluebenchmark.com/
"""
from lm_eval.api.task import PromptSourceTask


# TODO(jon-tow): Add citations for the individual datasets/tasks that make up GLUE.
_CITATION = """
@inproceedings{wang-etal-2018-glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355",
    abstract = "Human ability to understand language is \textit{general, flexible, and robust}. In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data. If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a unified model that can execute a range of linguistic tasks across different domains. To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE, gluebenchmark.com): a benchmark of nine diverse NLU tasks, an auxiliary dataset for probing models for understanding of specific linguistic phenomena, and an online platform for evaluating and comparing models. For some benchmark tasks, training data is plentiful, but for others it is limited or does not match the genre of the test set. GLUE thus favors models that can represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks. While none of the datasets in GLUE were created from scratch for the benchmark, four of them feature privately-held test data, which is used to ensure that the benchmark is used fairly. We evaluate baselines that use ELMo (Peters et al., 2018), a powerful transfer learning technique, as well as state-of-the-art sentence representation models. The best models still achieve fairly low absolute scores. Analysis with our diagnostic dataset yields similarly weak performance over all phenomena tested, with some exceptions.",
}
"""


# Single-Sentence Tasks


class CoLA(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "cola"

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


class SST(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "sst2"

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


# Inference Tasks


class MNLI(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test_matched"]


class MNLIMismatched(MNLI):
    VERSION = 0

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation_mismatched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test_mismatched"]


class QNLI(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qnli"

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


class WNLI(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "glue"
    DATASET_NAME = "wnli"

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


class RTE(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "rte"

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


# Similarity and Paraphrase Tasks


class MRPC(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mrpc"

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
        if (
            # generate_paraphrase for mrpc
            # This generation prompt assumes a positive example. We filter out the negative examples.
            # https://github.com/bigscience-workshop/promptsource/blob/ba8c9eccbe82f2409208c655896f1dd131171ece/promptsource/templates/glue/mrpc/templates.yaml#L7
            # https://github.com/bigscience-workshop/promptsource/blob/ba8c9eccbe82f2409208c655896f1dd131171ece/promptsource/templates/glue/mrpc/templates.yaml#L88
            (
                self.prompt_template.id == "3b88d2c4-0aeb-4c6d-9ccc-653a388250a5"
                or self.prompt_template.id == "d830d7a5-abc0-4275-ac62-974e0088876f"
            )
            and doc["label"] == 0
        ):
            return True
        return False


class QQP(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qqp"

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


class STSB(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "stsb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        return "sentence 1: {}\nsentence 2: {}\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["label"])

    def construct_requests(self, doc, ctx, args):
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def process_results(self, doc, results):
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def aggregation(self):
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")

    def higher_is_better(self):
        # TODO: implement evaluation.
        raise NotImplementedError("Evaluation not implemented")
