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

import numpy as np
from lm_eval.base import Task, MultipleChoiceTask, rf
from lm_eval.metrics import macro_f1_score, mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

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
        return map(self._process_doc,self.dataset["validation"])

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
        # 참 = entailment
        # 거짓 = contradiction
        # 무관 = neutral
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
