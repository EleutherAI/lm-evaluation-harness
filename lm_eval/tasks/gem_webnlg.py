"""
The 2020 Bilingual, Bi-Directional WebNLG+ Shared Task:
Overview and Evaluation Results (WebNLG+ 2020)
https://aclanthology.org/2020.webnlg-1.7/

WebNLG+ offers two challenges: (i) mapping sets of RDF triples
to English or Russian text (generation) and (ii) converting
English or Russian text to sets of RDF triples (semantic parsing).
Compared to the eponymous WebNLG challenge, WebNLG+ provides an
extended dataset that enable the training, evaluation, and
comparison of microplanners and semantic parsers. In this paper,
we present the results of the generation and semantic parsing
task for both English and Russian and provide a brief
description of the participating systems.
"""
from lm_eval.api.task import PromptSourceTask


_CITATION = """
@inproceedings{castro-ferreira-etal-2020-2020,
    title = "The 2020 Bilingual, Bi-Directional {W}eb{NLG}+ Shared Task: Overview and Evaluation Results ({W}eb{NLG}+ 2020)",
    author = "Castro Ferreira, Thiago  and
      Gardent, Claire  and
      Ilinykh, Nikolai  and
      van der Lee, Chris  and
      Mille, Simon  and
      Moussallem, Diego  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 3rd International Workshop on Natural Language Generation from the Semantic Web (WebNLG+)",
    month = "12",
    year = "2020",
    address = "Dublin, Ireland (Virtual)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.webnlg-1.7",
    pages = "55--76",
    abstract = "WebNLG+ offers two challenges: (i) mapping sets of RDF triples to English or Russian text (generation) and (ii) converting English or Russian text to sets of RDF triples (semantic parsing). Compared to the eponymous WebNLG challenge, WebNLG+ provides an extended dataset that enable the training, evaluation, and comparison of microplanners and semantic parsers. In this paper, we present the results of the generation and semantic parsing task for both English and Russian and provide a brief description of the participating systems.",
}
"""


class WebNLG(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "GEM/web_nlg"
    DATASET_NAME = "en"
    SPLIT = None

    def has_training_docs(self):
        return False

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
            if self.SPLIT is not None:
                return self.dataset[str(self.SPLIT)]
            else:
                return self.dataset["test"]

    def max_generation_length(self):
        return 250


class WebNLGRu(WebNLG):
    DATASET_NAME = "ru"


## En Challenge Sets


class WebNLGEn1(WebNLG):
    SPLIT = "challenge_validation_sample"


class WebNLGEn2(WebNLG):
    SPLIT = "challenge_test_scramble"


class WebNLGEn3(WebNLG):
    SPLIT = "challenge_test_numbers"


## Ru Challenge sets


class WebNLGRu1(WebNLG):
    DATASET_NAME = "ru"
    SPLIT = "challenge_validation_sample"


class WebNLGRu2(WebNLG):
    DATASET_NAME = "ru"
    SPLIT = "challenge_test_scramble"


WEBNLG_CLASSES = [
    WebNLG,
    WebNLGRu,
    WebNLGEn1,
    WebNLGEn2,
    WebNLGEn3,
    WebNLGRu1,
    WebNLGRu2,
]


def construct_tasks():
    tasks = {}
    for webnlg_class in WEBNLG_CLASSES:
        if webnlg_class.SPLIT is None:
            tasks[f"GEM/web_nlg_{webnlg_class.DATASET_NAME}"] = webnlg_class
        else:
            tasks[
                f"GEM/web_nlg_{webnlg_class.DATASET_NAME}_{webnlg_class.SPLIT}"
            ] = webnlg_class
    return tasks
