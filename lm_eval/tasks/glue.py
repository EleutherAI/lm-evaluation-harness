import numpy as np
from lm_eval.base import rf
from ..metrics import mean, matthews_corrcoef, f1_score
from . common import HFTask, yesno
from ..utils import general_detokenize

# Single-Sentence Tasks


class CoLA(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "cola"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        # TODO
        return ""

    def doc_to_text(self, doc):
        return "{}\nQuestion: Does this sentence make sense?\nAnswer:".format(doc["sentence"])

    def doc_to_target(self, doc):
        return " {}".format({1: "yes", 0: "no"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " yes")
        ll_false, _ = rf.loglikelihood(ctx, " no")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_true > ll_false
        gold = doc["label"]
        return {
            "mcc": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "mcc": True
        }

    def aggregation(self):
        return {
            "mcc": matthews_corrcoef
        }


class SST(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "sst2"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        return "Indicate if the sentiment of each sentence is positive or negative."

    def doc_to_text(self, doc):
        return "{}\nQuestion: Is this sentence positive or negative?\nAnswer:".format(
            general_detokenize(doc["sentence"]),
        )

    def doc_to_target(self, doc):
        return " {}".format({1: "positive", 0: "negative"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_positive, _ = rf.loglikelihood(ctx, " positive")
        ll_negative, _ = rf.loglikelihood(ctx, " negative")
        return ll_positive, ll_negative

    def process_results(self, doc, results):
        ll_positive, ll_negative = results
        pred = ll_positive > ll_negative
        gold = doc["label"]
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


# Inference Tasks


class MNLI(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation_matched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_matched"]

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["premise"],
            doc["hypothesis"].strip() + ('' if doc["hypothesis"].strip().endswith('.') else '.'),
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


class MNLIMismatched(MNLI):
    VERSION = 0

    def validation_docs(self):
        if self.has_validation_docs():
            return self.data["validation_mismatched"]

    def test_docs(self):
        if self.has_test_docs():
            return self.data["test_mismatched"]


class QNLI(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return "{}\n{}\nQuestion: Does this response answer the question?\nAnswer:".format(
            doc["question"],
            doc["sentence"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = not entailment
        return " {}".format({0: "yes", 1: "no"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        pred = ll_no > ll_yes
        gold = doc["label"]
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


class WNLI(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "wnli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True, False or Neither?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " {}".format({0: "True", 1: "Neither", 2: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


class RTE(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "rte"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def doc_to_text(self, doc):
        return "{}\nQuestion: {} True or False?\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        # 0 = entailment
        # 1 = not_entailment
        return " {}".format({0: "True", 1: "False"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_false

    def process_results(self, doc, results):
        ll_true, ll_false = results
        pred = ll_false > ll_true
        gold = doc["label"]
        return {
            "acc": pred == gold
        }

    def higher_is_better(self):
        return {
            "acc": True
        }

    def aggregation(self):
        return {
            "acc": mean
        }


# Similarity and Paraphrase Tasks


class MRPC(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "mrpc"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        return "Indicate if both sentences mean the same thing."

    def doc_to_text(self, doc):
        return "Sentence 1: {}\nSentence 2: {}\nQuestion: Do both sentences mean the same thing?\nAnswer:".format(
            general_detokenize(doc["sentence1"]),
            general_detokenize(doc["sentence2"]),
        )

    def doc_to_target(self, doc):
        return " {}".format(yesno(doc["label"]))

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "f1": f1_score
        }


class QQP(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "qqp"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def fewshot_description(self):
        return "Indicate if both questions ask the same thing."

    def doc_to_text(self, doc):
        return "Question 1: {}\nQuestion 2: {}\nQuestion: Do both questions ask the same thing?\nAnswer:".format(
            doc["question1"],
            doc["question2"],
        )

    def doc_to_target(self, doc):
        return " {}".format(yesno(doc["label"]))

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " yes")
        ll_no, _ = rf.loglikelihood(ctx, " no")
        return ll_yes, ll_no

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]
        pred = ll_yes > ll_no
        return {
            "acc": pred == gold,
            "f1": (gold, pred),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "f1": f1_score
        }


class STSB(HFTask):
    VERSION = 0
    DATASET_PATH = "glue"
    DATASET_NAME = "stsb"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def fewshot_description(self):
        return "Indicate if both sentences mean the same thing from a scale of 0-5, " \
           "where 5 means identical and 0 means unrelated."

    def doc_to_text(self, doc):
        return "sentence 1: {}\nsentence 2: {}\nAnswer:".format(
            doc["sentence1"],
            doc["sentence2"],
        )

    def doc_to_target(self, doc):
        return " {}".format(doc["label"])

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of 
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural 
            language description, as well as the few shot examples, and the question
            part of the document for `doc`. 
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        # TODO: implement evaluation.
        raise NotImplementedError('Evaluation not implemented')
