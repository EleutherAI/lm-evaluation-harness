"""
WMT: Workshop on Statistical Machine Translation

WMT is the main event for machine translation and machine translation research.
The conference is held annually in connection with larger conferences on natural
language processing.

Homepage: https://machinetranslate.org/wmt
"""
import promptsource.utils
from typing import Dict, List, Optional

from lm_eval.api.task import TranslationTask


# TODO: Add each WMT year BibTeX citation.
_CITATION = """
"""


class WMTBase(TranslationTask):
    VERSION = 0

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

    def max_generation_length(self) -> Optional[int]:
        return 64


def _year_to_lang_pairs(start_year: int, end_year: int) -> Dict[str, List[str]]:
    """Downloads the config names for WMT years and returns a dict of language
    language pairs for each such year.
    """
    year_to_lang_pairs = {}
    wmtyears = [f"wmt{year}" for year in range(start_year, end_year + 1)]
    for wmtyear in wmtyears:
        year_to_lang_pairs[wmtyear] = [
            c.name for c in promptsource.utils.get_dataset_confs(wmtyear)
        ]
    return year_to_lang_pairs


# Hard-code `_year_to_lang_pairs(14, 19)` to avoid downloading configs every time.
_YEAR_TO_LANG_PAIRS = {
    "wmt14": ["cs-en", "de-en", "fr-en", "hi-en", "ru-en"],
    "wmt15": ["cs-en", "de-en", "fi-en", "fr-en", "ru-en"],
    "wmt16": ["cs-en", "de-en", "fi-en", "ro-en", "ru-en", "tr-en"],
    "wmt17": ["cs-en", "de-en", "fi-en", "lv-en", "ru-en", "tr-en", "zh-en"],
    "wmt18": ["cs-en", "de-en", "et-en", "fi-en", "kk-en", "ru-en", "tr-en", "zh-en"],
    "wmt19": ["cs-en", "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en", "fr-de"],  # fmt: skip
}


def construct_tasks() -> Dict[str, WMTBase]:
    """Constructs a `dict` of WMT tasks for all available WMT years
    with keys of the form:
        `wmt{year}_{lang1}_{lang2}`
    Example:
        `wmt14_cs_en`
    """
    tasks = {}
    for wmtyear, lang_pairs in _YEAR_TO_LANG_PAIRS.items():
        for lang_pair in lang_pairs:
            task_class = _create_wmt_class(dataset_path=wmtyear, dataset_name=lang_pair)
            lang_pair = lang_pair.replace("-", "_")
            tasks[f"{wmtyear}_{lang_pair}"] = task_class
    return tasks


def _create_wmt_class(
    dataset_path: str, dataset_name: str, version: Optional[int] = 0
) -> WMTBase:
    class WMT(WMTBase):
        DATASET_PATH = dataset_path
        DATASET_NAME = dataset_name

    return WMT
