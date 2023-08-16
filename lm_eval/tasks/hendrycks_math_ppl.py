"""
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/pdf/2103.03874.pdf

Math is a dataset of 12,500 challenging competition mathematics problems. Each
problem in Math has a full step-by-step solution which can be used to teach
models to generate answer derivations and explanations.

Homepage: https://github.com/hendrycks/math
"""
import inspect
import lm_eval.datasets.hendrycks_math.hendrycks_math
from lm_eval.metrics import mean, weighted_perplexity
from lm_eval.base import Task, rf


_CITATION = """
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the Math Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
"""


class Math(Task):
    DATASET_PATH = inspect.getfile(lm_eval.datasets.hendrycks_math.hendrycks_math)
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        return map(self._process_doc, self.dataset["train"])

    def validation_docs(self):
        return NotImplemented

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return "Problem: " + doc["problem"] + "\nAnswer:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["problem"]

    def doc_to_target(self, doc):
        return " " + doc["solution"] + " The answer is " + self.last_boxed_only_string(doc['solution']) + ".```"
    
    def last_boxed_only_string(self, string):

        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None

        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx : right_brace_idx + 1]

        return retval

    def construct_requests(self, doc, ctx, params={}):
        ll, _ = rf.loglikelihood(ctx, self.doc_to_target(doc))
        return ll

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        ll = results[0]
        bytes_ = self.count_bytes(doc)
        return {"ppl": (ll, bytes_)}
    
    def count_bytes(self, doc):
        return len(self.doc_to_target(doc).encode("utf-8"))

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"ppl": weighted_perplexity}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"ppl": False}

class MathAlgebraEasy(Math):
    VERSION = 1
    DATASET_NAME = "algebra"

    def training_docs(self):
        data = map(self._process_doc, self.dataset["train"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data

    def test_docs(self):
        data = map(self._process_doc, self.dataset["train"])
        data = filter(lambda x: x['level'] == 'Level 1', data)
        return data


class MathAlgebra(Math):
    VERSION = 1
    DATASET_NAME = "algebra"


class MathCountingAndProbability(Math):
    VERSION = 1
    DATASET_NAME = "counting_and_probability"


class MathGeometry(Math):
    VERSION = 1
    DATASET_NAME = "geometry"


class MathIntermediateAlgebra(Math):
    VERSION = 1
    DATASET_NAME = "intermediate_algebra"


class MathNumberTheory(Math):
    VERSION = 1
    DATASET_NAME = "number_theory"


class MathPrealgebra(Math):
    VERSION = 1
    DATASET_NAME = "prealgebra"


class MathPrecalculus(Math):
    VERSION = 1
    DATASET_NAME = "precalculus"
