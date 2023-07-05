from functools import partial
import json
from math import exp
import os

from ..squad import _squad_agg
from lm_eval.base import Task, rf


"""
    FQuAD: French Question Answering Dataset
    https://fquad.illuin.tech/publication/fquad/
    (note: cannot be used for commercial use or training but fine for eval purposes)
"""


class FQuAD(Task):
    VERSION = 0

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.load_docs("validation")

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        # access to the dataset has to be requested with the fquad authors
        # and this folder has to be manually created from the provided files
        self.data_file_path = os.environ["PATH_TO_FQUAD"]

        assert os.path.exists(
            self.data_file_path + "valid.json"
        ), f"Data file expected in: {self.data_file_path + 'valid.json'}"

    def load_docs(self, split):
        filename = "valid.json" if split == "validation" else "train.json"
        with open(self.data_file_path + filename, "r", encoding="UTF-8") as f:
            data = json.load(f)
            id = -1
            for item in data["data"]:
                for paragraph in item["paragraphs"]:
                    for qas in paragraph["qas"]:
                        id += 1
                        yield {
                            "id": id,
                            "context": paragraph["context"],
                            "question": qas["question"],
                            "answers": {
                                "text": [
                                    qas["answers"][i]["text"]
                                    for i in range(len(qas["answers"]))
                                ],
                                "answer_start": [
                                    qas["answers"][i]["answer_start"]
                                    for i in range(len(qas["answers"]))
                                ],
                            },
                        }

    def fewshot_description(self):
        return (
            "En tenant compte du contexte, répondez succinctement à la question posée."
        )

    def doc_to_text(self, doc):
        return (
            "Contexte: "
            + doc["context"]
            + "\nQuestion:"
            + doc["question"]
            + "\nRéponse:"
        )

    def doc_to_target(self, doc):
        answer_list = doc["answers"]["text"]
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = "unanswerable"
        return " " + answer

    def construct_requests(self, doc, ctx):
        continuation = rf.greedy_until(ctx, {"until": ["\n"]})
        is_unanswerable = rf.loglikelihood(ctx, " " + "unanswerable")
        return continuation, is_unanswerable

    def process_results(self, doc, results):
        continuation, (logprob_unanswerable, _) = results
        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
            "no_answer_probability": no_answer_probability,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
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
        }

    def aggregation(self):
        return {
            "exact": partial(
                _squad_agg, "exact"
            ),  # Exact match (the normalized answer exactly match the gold answer)
            "f1": partial(
                _squad_agg, "f1"
            ),  # The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,  # The F-score of predicted tokens versus the gold answer
        }
