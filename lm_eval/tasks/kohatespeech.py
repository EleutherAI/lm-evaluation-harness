"""
For hate speech, they introduce hate, offensive, and none labels.
They also added binary label whether a comment contains gender bias or not.
https://aclanthology.org/2020.socialnlp-1.4.pdf

Updated on May 06 2023
APEACH is the first crowd-generated Korean evaluation dataset for hate speech detection. 
"""

import numpy as np
from lm_eval.base import Task, MultipleChoiceTask, rf
from lm_eval.metrics import macro_f1_score, mean, matthews_corrcoef, f1_score, yesno
from lm_eval.utils import general_detokenize

_CITATION1 ="""
@inproceedings{moon-etal-2020-beep,
    title = "{BEEP}! {K}orean Corpus of Online News Comments for Toxic Speech Detection",
    author = "Moon, Jihyung  and
      Cho, Won Ik  and
      Lee, Junbum",
    booktitle = "Proceedings of the Eighth International Workshop on Natural Language Processing for Social Media",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.socialnlp-1.4",
    pages = "25--31",
    abstract = "Toxic comments in online platforms are an unavoidable social issue under the cloak of anonymity. Hate speech detection has been actively done for languages such as English, German, or Italian, where manually labeled corpus has been released. In this work, we first present 9.4K manually labeled entertainment news comments for identifying Korean toxic speech, collected from a widely used online news platform in Korea. The comments are annotated regarding social bias and hate speech since both aspects are correlated. The inter-annotator agreement Krippendorff{'}s alpha score is 0.492 and 0.496, respectively. We provide benchmarks using CharCNN, BiLSTM, and BERT, where BERT achieves the highest score on all tasks. The models generally display better performance on bias identification, since the hate speech detection is a more subjective issue. Additionally, when BERT is trained with bias label for hate speech detection, the prediction score increases, implying that bias and hate are intertwined. We make our dataset publicly available and open competitions with the corpus and benchmarks.",
}
"""

class HateSpeech(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "cardy/kohatespeech"
    DATASET_NAME = "hate_speech"
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return doc["query"]

    def _process_doc(self, doc):
        out_doc = {
            "query": "문장: {}".format(doc["comments"]),
            "choices": ["없음", "공격적", "혐오"], # ["none", "offensive", "hate"]
            "gold": doc['hate']
        }
        return out_doc

    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["gold"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }


class GenderBias(Task):
    VERSION = 0
    DATASET_PATH = "cardy/kohatespeech"
    DATASET_NAME = "hate_speech"
    
    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self._training_docs is None:
            self._training_docs = list(map(self._process_doc, self.dataset["train"]))
        return self._training_docs

    def validation_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return "문장: {} 성적 편향이 있습니까?".format(doc["comments"])

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"}[doc["contain_gender_bias"]])

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")
        return ll_no, ll_yes
    
    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["contain_gender_bias"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }

_CITATION2 = """
@inproceedings{yang-etal-2022-apeach,
    title = "{APEACH}: Attacking Pejorative Expressions with Analysis on Crowd-Generated Hate Speech Evaluation Datasets",
    author = "Yang, Kichang  and
      Jang, Wonjun  and
      Cho, Won Ik",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.525",
    pages = "7076--7086",
    abstract = "In hate speech detection, developing training and evaluation datasets across various domains is the critical issue. Whereas, major approaches crawl social media texts and hire crowd-workers to annotate the data. Following this convention often restricts the scope of pejorative expressions to a single domain lacking generalization. Sometimes domain overlap between training corpus and evaluation set overestimate the prediction performance when pretraining language models on low-data language. To alleviate these problems in Korean, we propose APEACH that asks unspecified users to generate hate speech examples followed by minimal post-labeling. We find that APEACH can collect useful datasets that are less sensitive to the lexical overlaps between the pretraining corpus and the evaluation set, thereby properly measuring the model performance.",
}
"""


class Apeach(Task):
    VERSION = 0
    DATASET_PATH = "jason9693/APEACH" 
    DATASET_NAME = "apeach"
    
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def doc_to_text(self, doc):
        return "문장: {} 혐오 발화가 있습니까?".format(doc["text"])

    def doc_to_target(self, doc):
        return " {}".format({0: "아니오", 1: "예"}[doc["class"]])

    def construct_requests(self, doc, ctx):
        ll_no, _ = rf.loglikelihood(ctx, " 아니오")
        ll_yes, _ = rf.loglikelihood(ctx, " 예")
        return ll_no, ll_yes
    
    def process_results(self, doc, results):
        pred = np.argmax(results)
        gold = doc["class"]
        return {
            "acc": pred == gold,
            "macro_f1": (gold, pred)
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "macro_f1": True
        }

    def aggregation(self):
        return {
            "acc": mean,
            "macro_f1": macro_f1_score
        }
    