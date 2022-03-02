"""
TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension
https://arxiv.org/pdf/1705.03551.pdf

TriviaQA is a reading comprehension dataset containing over 650K question-answer-evidence
triples. TriviaQA includes 95K question-answer pairs authored by trivia enthusiasts
and independently gathered evidence documents, six per question on average, that provide
high quality distant supervision for answering the questions.

Homepage: https://nlp.cs.washington.edu/triviaqa/
"""
import os
import json
import jsonlines
from lm_eval.base import Task, rf
from ..metrics import mean
from ..utils import sh
from best_download import download_file


_CITATION = """
@InProceedings{JoshiTriviaQA2017,
    author = {Joshi, Mandar and Choi, Eunsol and Weld, Daniel S. and Zettlemoyer, Luke},
    title = {TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension},
    booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
    month = {July},
    year = {2017},
    address = {Vancouver, Canada},
    publisher = {Association for Computational Linguistics},
}
"""


class TriviaQA(Task):
    VERSION = 0
    def download(self):
        if not os.path.exists('data/triviaqa/unfiltered-web-train.jsonl'):
            os.makedirs("data/triviaqa/", exist_ok=True)
            download_file("http://eaidata.bmk.sh/data/triviaqa-unfiltered.tar.gz", local_file="data/triviaqa/triviaqa-unfiltered.tar.gz", expected_checksum="adc19b42769062d241a8fbe834c56e58598d9322eb6c614e9f33a68a2cf5523e")
            sh("""
            cd data/triviaqa/
            tar -xf triviaqa-unfiltered.tar.gz
            """)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-train.jsonl')

    def validation_docs(self):
        return jsonlines.open('data/triviaqa/unfiltered-web-dev.jsonl')

    def test_docs(self):
        raise NotImplementedError()
    
    def doc_to_text(self, doc):
        return f"Question: {doc['Question']}\nAnswer:"

    def doc_to_target(self, doc):
        return " " + doc['Answer']['Value']

    def _remove_prefixes(self, aliases):
        # Optimization: Remove any alias that has a strict prefix elsewhere in the list
        # we can do this because if the prefix is acceptable by isgreedy, we can stop looking
        aliases.sort()
        ret = [aliases[0]]
        for alias in aliases[1:]:
            if not alias.startswith(ret[-1]):
                ret.append(alias)

        return ret

    def construct_requests(self, doc, ctx):
        ret = []
        for alias in self._remove_prefixes(doc['Answer']['Aliases']):
            _, is_prediction = rf.loglikelihood(ctx, " " + alias)
            ret.append(is_prediction)
        return ret

    def process_results(self, doc, results):
        return {
            "acc": float(any(results))
        }

    def aggregation(self):
        return {
            "acc": mean,
        }

    def higher_is_better(self):
        return {
            "acc": True
        }
