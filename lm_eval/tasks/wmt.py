"""
WMT: Workshop on Statistical Machine Translation

WMT is the main event for machine translation and machine translation research.
The conference is held annually in connection with larger conferences on natural
language processing.

Homepage: https://machinetranslate.org/wmt
"""
import typing

from lm_eval.api.task import TranslationTask


# TODO: Add each WMT year BibTeX citation.
_CITATION = """
"""


class WMTBase(TranslationTask):
    def training_docs(self):
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self) -> typing.Optional[int]:
        return 64


# WMT 2014


class WMT14Base(WMTBase):
    DATASET_PATH = "wmt14"

    def has_training_docs(self):
        return True

    def has_test_docs(self):
        return True

    def has_validation_docs(self):
        return True


class WMT14FrEn(WMT14Base):
    VERSION = 0
    DATASET_NAME = "fr-en"


class WMT14DeEn(WMT14Base):
    VERSION = 0
    DATASET_NAME = "de-en"


# TODO: Add more language pairs when `promptsource` is updated.
WMT14_TASKS = [WMT14FrEn, WMT14DeEn]


def create_year_tasks(year_classes) -> typing.Dict[str, WMTBase]:
    """
    Utility for creating a `dict` of WMT tasks for a specific year.

    Args:
        year_classes: A list of task classes for a given WMT year.
            NOTE: Use only the task classes defined in this file,
            e.g. `WMT14_TASKS`.
    """
    tasks = {}
    for task_class in year_classes:
        benchmark = task_class.DATASET_PATH
        lang_pair = task_class.DATASET_NAME.replace("-", "_")
        tasks[f"{benchmark}_{lang_pair}"] = task_class
    return tasks
