# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferrably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.
"""

from lm_eval.base import PromptSourceTask
from lm_eval.base import Task, rf


_CITATION = """ """

class GEMWikiLinguaBase(PromptSourceTask):
    VERSION = 0
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
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs
    
    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]
    
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
