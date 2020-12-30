import json
import random
from lm_eval.base import Dataset
from ..utils import sh
from .common import simple_accuracy_metric
import csv
import os


class StoryCloze(Dataset):
    def download(self):
        # TODO: replace with Eye link
        pass

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    @staticmethod
    def _assert_has_data():
        if not os.path.exists("data/storycloze"):
            raise AssertionError(
                'StoryCloze must be downloaded manually. Please download the dataset and store it in "data/storycloze/".'
            )

    def training_docs(self):
        self._assert_has_data()
        # StoryCloze has no training data, so following the GPT-3 paper we will generate few-shot examples using the development set,
        # and evaluate using the test set
        return self.load_doc(
            "data/storycloze/cloze_test_val__spring2016 - cloze_test_ALL_val.csv"
        )

    def load_doc(self, filename):
        with open(filename) as f:
            return list(csv.DictReader(f))

    def validation_docs(self):
        self._assert_has_data()
        # StoryCloze has no training data, so following the GPT-3 paper we will generate few-shot examples using the development set,
        # and evaluate using the test set
        return self.load_doc(
            "data/storycloze/cloze_test_test__spring2016 - cloze_test_ALL_test.csv"
        )

    def test_docs(self):
        self._assert_has_data()
        return self.load_doc(
            "data/storycloze/cloze_test_test__spring2016 - cloze_test_ALL_test.csv"
        )

    def fewshot_description(self):
        pass

    def doc_to_text(self, doc, include_target=True):
        storycloze_prompt = "{} {} {} {}".format(
            doc["InputSentence1"],
            doc["InputSentence2"],
            doc["InputSentence3"],
            doc["InputSentence4"],
        )

        if include_target:
            if doc["AnswerRightEnding"] == "1":
                return storycloze_prompt + " " + doc["RandomFifthSentenceQuiz1"]
            else:
                return storycloze_prompt + " " + doc["RandomFifthSentenceQuiz2"]
        else:
            return storycloze_prompt

    def evaluate(self, docs, lm, provide_description, num_fewshot):
        golds = [doc["AnswerRightEnding"] for doc in docs]
        preds = []
        for doc in docs:
            ctx = self.fewshot_context(
                doc=doc,
                provide_description=provide_description,
                num_fewshot=num_fewshot,
            )
            loglikelihood_with_sentence_1 = lm.loglikelihood(
                ctx, " " + doc["RandomFifthSentenceQuiz1"]
            ).sum()
            loglikelihood_with_sentence_2 = lm.loglikelihood(
                ctx, " " + doc["RandomFifthSentenceQuiz2"]
            ).sum()
            if loglikelihood_with_sentence_1 > loglikelihood_with_sentence_2:
                preds.append("1")
            else:
                preds.append("2")

        return simple_accuracy_metric(preds=preds, golds=golds)
