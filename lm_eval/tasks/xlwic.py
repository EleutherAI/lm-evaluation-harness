"""
XL-WiC: A Multilingual Benchmark for Evaluating Semantic Contextualization
https://arxiv.org/abs/2010.06478

We put forward a large multilingual benchmark, XL-WiC, featuring gold standards in 12 new languages from varied language families
and with different degrees of resource availability, opening room for evaluation scenarios such as zero-shot cross-lingual
transfer. XL-WiC is framed as a binary classification task. Each instance in XL-WiC has a target word w,
either a verb or a noun, for which two contexts are provided. Each of these contexts triggers a
specific meaning of w. The task is to identify if the occurrences of w in the two contexts correspond to the same meaning or not.

Homepage: https://pilehvar.github.io/xlwic/

"""
from lm_eval.base import rf, Task
from lm_eval.metrics import mean

_CITATION = """
@article{DBLP:journals/corr/abs-2010-06478,
  author       = {Alessandro Raganato and
                  Tommaso Pasini and
                  Jos{\'{e}} Camacho{-}Collados and
                  Mohammad Taher Pilehvar},
  title        = {XL-WiC: {A} Multilingual Benchmark for Evaluating Semantic Contextualization},
  journal      = {CoRR},
  volume       = {abs/2010.06478},
  year         = {2020},
  url          = {https://arxiv.org/abs/2010.06478},
  eprinttype    = {arXiv},
  eprint       = {2010.06478},
  timestamp    = {Tue, 20 Oct 2020 15:08:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2010-06478.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
"""


class WordsInContextBase(Task):
    VERSION = 0
    DATASET_PATH = "pasinit/xlwic"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results):
        ll_yes, ll_no = results
        gold = doc["label"]

        acc = 1.0 if (ll_yes > ll_no) == gold else 0.0

        return {"acc": acc}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class WordsInContextIt(WordsInContextBase):
    DATASET_NAME = "xlwic_it_it"

    def doc_to_text(self, doc):
        return (
            "Frase 1: {}\nFrase 2: {}\nDomande: La parola '{}' è usata allo stesso modo nelle due frasi precedenti?"
            "\nRisposta:".format(
                doc["context_1"],
                doc["context_2"],
                doc["target_word"],
            )
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "no", 1: "sì"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " sì")
        ll_no, _ = rf.loglikelihood(ctx, " no")

        return ll_yes, ll_no


class WordsInContextDe(WordsInContextBase):
    DATASET_NAME = "xlwic_de_de"

    def doc_to_text(self, doc):
        return (
            "Satz 1: {}\nSatz 2: {}\nFrage: Wird das Wort '{}' in den beiden obigen Sätzen auf dieselbe Weise verwendet?"
            "\nAntwort:".format(
                doc["context_1"],
                doc["context_2"],
                doc["target_word"],
            )
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "nein", 1: "ja"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_yes, _ = rf.loglikelihood(ctx, " ja")
        ll_no, _ = rf.loglikelihood(ctx, " nein")

        return ll_yes, ll_no
