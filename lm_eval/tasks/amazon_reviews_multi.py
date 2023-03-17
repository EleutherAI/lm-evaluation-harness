# It was based on gem_wikilingua.py

from lm_eval.api.task import PromptSourceTask
import typing


class AmazonReviewsMultiBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "amazon_reviews_multi"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    # TODO: Add max length to improve perf (I guess)
    # def max_generation_length(self):
    #     return 64


class AmazonReviewsMultiFr(AmazonReviewsMultiBase):
    DATASET_NAME = "fr"


class AmazonReviewsMultiEn(AmazonReviewsMultiBase):
    DATASET_NAME = "en"


AMAZON_REVIEWS_TASKS = [
    AmazonReviewsMultiFr,
    AmazonReviewsMultiEn,
]


def construct_tasks() -> typing.Dict[str, AmazonReviewsMultiBase]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "amazon_reviews_multi_fr"
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in AMAZON_REVIEWS_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks
