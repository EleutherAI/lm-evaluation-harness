"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053

@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""


import numpy as np
from lm_eval.base import rf
from ..metrics import mean
from . common import HFTask

LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]


class XNLIBase(HFTask):
    VERSION = 0
    DATASET_PATH = "xnli"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return doc['premise'] + '\nQuestion: ' + doc['hypothesis'] + ' True, False, or Neither?\nAnswer:'

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " " + ["True", "Neither", "False"][doc['label']]

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
        ll_true, _ = rf.loglikelihood(ctx, " True")
        ll_neither, _ = rf.loglikelihood(ctx, " Neither")
        ll_false, _ = rf.loglikelihood(ctx, " False")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["label"]
        pred = np.argmax(results)
        return {
            "acc": pred == gold
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "acc": True
        }


class XNLI_ar(XNLIBase):
    DATASET_NAME = "ar"

class XNLI_bg(XNLIBase):
    DATASET_NAME = "bg"

class XNLI_de(XNLIBase):
    DATASET_NAME = "de"

class XNLI_el(XNLIBase):
    DATASET_NAME = "el"

class XNLI_en(XNLIBase):
    DATASET_NAME = "en"

class XNLI_es(XNLIBase):
    DATASET_NAME = "es"

class XNLI_fr(XNLIBase):
    DATASET_NAME = "fr"

class XNLI_hi(XNLIBase):
    DATASET_NAME = "hi"

class XNLI_ru(XNLIBase):
    DATASET_NAME = "ru"

class XNLI_sw(XNLIBase):
    DATASET_NAME = "sw"

class XNLI_th(XNLIBase):
    DATASET_NAME = "th"

class XNLI_tr(XNLIBase):
    DATASET_NAME = "tr"

class XNLI_ur(XNLIBase):
    DATASET_NAME = "ur"

class XNLI_vi(XNLIBase):
    DATASET_NAME = "vi"

class XNLI_zh(XNLIBase):
    DATASET_NAME = "zh"


LANG_CLASSES = [
    XNLI_ar, XNLI_bg, XNLI_de, XNLI_el, XNLI_en, XNLI_es, XNLI_fr, XNLI_hi,
    XNLI_ru, XNLI_sw, XNLI_th, XNLI_tr, XNLI_ur, XNLI_vi, XNLI_zh
]

def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks
