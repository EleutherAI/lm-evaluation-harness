"""
WikiLingua: A New Benchmark Dataset for Cross-Lingual Abstractive Summarization
https://arxiv.org/pdf/2010.03093.pdf

Wikilingua is a large-scale (~770k article-summary pairs), multilingual dataset for the evaluation of cross-lingual abstractive systems.
It consists of parallel articles and summaries (article-summary pairs) from WikiHow across 18 languages (i.e. all the languages available on WikiHow).
It contains 141,457 unique English articles and each of the other 17 languages has on average, 42,783 articles that align with an article in English.
This dataset is part of the GEM Benchmark. (Description from https://gem-benchmark.com/data_cards/WikiLingua)


Homepage: None, Repo: https://github.com/esdurmus/Wikilingua
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{ladhak-wiki-2020,
    title={WikiLingua: A New Benchmark Dataset for Multilingual Abstractive Summarization},
    author={Faisal Ladhak, Esin Durmus, Claire Cardie and Kathleen McKeown},
    booktitle={Findings of EMNLP, 2020},
    year={2020}
}"""


class GEMWikiLinguaBase(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "GEM/wiki_lingua"
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
            return self.dataset["sampled_test"]

    def max_generation_length(self):
        return 64


class GEMWikiLinguaAr(GEMWikiLinguaBase):
    DATASET_NAME = "ar"


class GEMWikiLinguaCs(GEMWikiLinguaBase):
    DATASET_NAME = "cs"


class GEMWikiLinguaDe(GEMWikiLinguaBase):
    DATASET_NAME = "de"


class GEMWikiLinguaEn(GEMWikiLinguaBase):
    DATASET_NAME = "en"


class GEMWikiLinguaEs(GEMWikiLinguaBase):
    DATASET_NAME = "es"


class GEMWikiLinguaFr(GEMWikiLinguaBase):
    DATASET_NAME = "fr"


class GEMWikiLinguaHi(GEMWikiLinguaBase):
    DATASET_NAME = "hi"


class GEMWikiLinguaId(GEMWikiLinguaBase):
    DATASET_NAME = "id"


class GEMWikiLinguaIt(GEMWikiLinguaBase):
    DATASET_NAME = "it"


class GEMWikiLinguaJa(GEMWikiLinguaBase):
    DATASET_NAME = "ja"


class GEMWikiLinguaKo(GEMWikiLinguaBase):
    DATASET_NAME = "ko"


class GEMWikiLinguaNl(GEMWikiLinguaBase):
    DATASET_NAME = "nl"


class GEMWikiLinguaPt(GEMWikiLinguaBase):
    DATASET_NAME = "pt"


class GEMWikiLinguaRu(GEMWikiLinguaBase):
    DATASET_NAME = "ru"


class GEMWikiLinguaTh(GEMWikiLinguaBase):
    DATASET_NAME = "th"


class GEMWikiLinguaTr(GEMWikiLinguaBase):
    DATASET_NAME = "tr"


class GEMWikiLinguaVi(GEMWikiLinguaBase):
    DATASET_NAME = "vi"


class GEMWikiLinguaZh(GEMWikiLinguaBase):
    DATASET_NAME = "zh"


WIKILINGUA_TASKS = [
    GEMWikiLinguaAr,
    GEMWikiLinguaCs,
    GEMWikiLinguaDe,
    GEMWikiLinguaEn,
    GEMWikiLinguaEs,
    GEMWikiLinguaFr,
    GEMWikiLinguaHi,
    GEMWikiLinguaId,
    GEMWikiLinguaIt,
    GEMWikiLinguaJa,
    GEMWikiLinguaKo,
    GEMWikiLinguaNl,
    GEMWikiLinguaPt,
    GEMWikiLinguaRu,
    GEMWikiLinguaTh,
    GEMWikiLinguaTr,
    GEMWikiLinguaVi,
    GEMWikiLinguaZh,
]


def construct_tasks() -> typing.Dict[str, GEMWikiLinguaBase]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "GEM/wiki_lingua_ar"
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in WIKILINGUA_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks
