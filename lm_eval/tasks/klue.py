"""
KLUE
https://arxiv.org/abs/2105.09680

 Korean Language Understanding Evaluation (KLUE) benchmark is a series of datasets
 to evaluate natural language understanding capability of Korean language models.
 KLUE consists of 8 diverse and representative tasks, which are accessible to anyone without any restrictions.
 With ethical considerations in mind, we deliberately design annotation guidelines
 to obtain unambiguous annotations for all datasets. Furthermore, we build an evaluation system
 and carefully choose evaluations metrics for every task, thus establishing fair comparison across Korean language models.
 
 Homepage: https://klue-benchmark.com/
"""

import datasets
import evaluate
from math import exp
import numpy as np
from lm_eval.base import Task, MultipleChoiceTask, rf
from lm_eval.metrics import macro_f1_score, mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize
from functools import partial

_CITATION = """
@misc{park2021klue,
      title={KLUE: Korean Language Understanding Evaluation},
      author={Sungjoon Park and Jihyung Moon and Sungdong Kim and Won Ik Cho and Jiyoon Han and Jangwon Park and Chisung Song and Junseong Kim and Yongsook Song and Taehwan Oh and Joohong Lee and Juhyun Oh and Sungwon Lyu and Younghoon Jeong and Inkwon Lee and Sangwoo Seo and Dongjun Lee and Hyunwoo Kim and Myeonghwa Lee and Seongbo Jang and Seungwon Do and Sunkyoung Kim and Kyungtae Lim and Jongwon Lee and Kyumin Park and Jamin Shin and Seonghyun Kim and Lucy Park and Alice Oh and Jungwoo Ha and Kyunghyun Cho},
      year={2021},
      eprint={2105.09680},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


def _klue_mrc_metric(predictions, references):
    klue_mrc_metric = evaluate.load("ingyu/klue_mrc")

    return klue_mrc_metric.compute(predictions=predictions, references=references)


def _klue_mrc_agg(key, items):
    predictions, references = zip(*items)

    return _klue_mrc_metric(predictions=predictions, references=references)[key]


class STS(Task):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "sts"
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "질문: 문장 1과 문장 2는 서로 유사한 의미를 가지나요?\n문장 1: {}\n문장 2: {}\n정답:".format(
            general_detokenize(doc["sentence1"]),
            general_detokenize(doc["sentence2"]) 
        )

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"}[doc["labels"]["binary-label"]])

    def construct_requests(self, doc, ctx):
        ll_negative, _ = rf.loglikelihood(ctx, " 아니오")
        ll_positive, _ = rf.loglikelihood(ctx, " 예")
        return ll_negative, ll_positive

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["labels"]["binary-label"]
        return {
            "acc": pred == gold,
            "f1": (gold, pred)
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


class YNAT(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "ynat"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc,self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "title": doc["title"],
            "choices": ["과학", "경제", "사회", "생활", "세계", "스포츠", "정치"],
            "gold": doc["label"]
        }
        return out_doc

    def doc_to_text(self, doc):
        return "{}".format(doc["title"])

    def doc_to_target(self, doc):
        return " ({})".format({0: "과학", 1: "경제", 2: "사회", 3: "생활", 4: "세계", 5: "스포츠", 6: "정치"}[doc["gold"]])

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]
        return {
            "f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "f1": True
        }

    def aggregation(self):
        return {
            "f1": macro_f1_score
        }


class NLI(Task):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "nli"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return "{}\n질문: {} 참, 거짓, 중립 중 무엇인가요?\n정답:".format(
            doc["premise"],
            doc["hypothesis"].strip()
            + ("" if doc["hypothesis"].strip().endswith(".") else "."),
        )

    def doc_to_target(self, doc):
        """
        참 = entailment
        거짓 = contradiction
        무관 = neutral
        """
        return " {}".format({0: "참", 1: "중립", 2: "거짓"}[doc["label"]])

    def construct_requests(self, doc, ctx):
        ll_true, _ = rf.loglikelihood(ctx, " 참")
        ll_neither, _ = rf.loglikelihood(ctx, " 중립")
        ll_false, _ = rf.loglikelihood(ctx, " 거짓")
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def higher_is_better(self):
        return {"acc": True}

    def aggregation(self):
        return {"acc": mean}


class MRC(Task):
    VERSION = 0
    DATASET_PATH = "klue"
    DATASET_NAME = "mrc"

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

    def doc_to_text(self, doc):
        return "제목: " + doc["title"] + "\n\n" + "본문: " + doc["context"] + "\n\n" + "질문: " + doc["question"] + "\n\n" + "답:"

    def doc_to_target(self, doc):
        answer = doc["answers"]["text"][0]
        if doc["is_impossible"]:
            answer = "대답 불가"
        return " " + answer

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
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        is_unanswerable = rf.loglikelihood(ctx, " " + "대답 불가")
        return continuation, is_unanswerable
    
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a 
        dict where keys are the names of submetrics and values are the values of 
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        no_answer_probability = exp(logprob_unanswerable)
        
        predictions = {
            'id': doc['guid'],
            'prediction_text': continuation,
            'no_answer_probability': no_answer_probability,
        }

        references = {
            'id': doc['guid'],
            'answers': doc['answers'],
            'unanswerable': doc['is_impossible'],
        }

        return {
            "exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": (
                predictions,
                references,
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": (
                predictions,
                references,
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": (
                predictions,
                references,
            ),  # Best exact match (with varying threshold)
            "best_f1": (predictions, references),  # Best F1 (with varying threshold)
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are 
            functions that aggregate a list of metrics
        """
        return {
            "exact": partial(
                _klue_mrc_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _klue_mrc_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": partial(
                _klue_mrc_agg, "HasAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": partial(
                _klue_mrc_agg, "HasAns_f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": partial(
                _klue_mrc_agg, "NoAns_exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": partial(
                _klue_mrc_agg, "NoAns_f1"
            ),  # The F-score of predicted tokens versus the gold answer
            "best_exact": partial(
                _klue_mrc_agg, "best_exact"
            ),  # Best exact match (with varying threshold)
            "best_f1": partial(
                _klue_mrc_agg, "best_f1"
            ),  # Best F1 (with varying threshold)
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are 
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
            "HasAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "HasAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "NoAns_exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "NoAns_f1": True,  # The F-score of predicted tokens versus the gold answer
            "best_exact": True,  # Best exact match (with varying threshold)
            "best_f1": True,  # Best F1 (with varying threshold)
        }
