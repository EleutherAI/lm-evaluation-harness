"""
XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning
https://ducdauge.github.io/files/xcopa.pdf

The Cross-lingual Choice of Plausible Alternatives dataset is a benchmark to evaluate the ability of machine learning models to transfer commonsense reasoning across languages.
The dataset is the translation and reannotation of the English COPA (Roemmele et al. 2011) and covers 11 languages from 11 families and several areas around the globe.
The dataset is challenging as it requires both the command of world knowledge and the ability to generalise to new languages.
All the details about the creation of XCOPA and the implementation of the baselines are available in the paper.

Homepage: https://github.com/cambridgeltl/xcopa
"""
from .superglue import Copa


_CITATION = """
@inproceedings{ponti2020xcopa,
  title={{XCOPA: A} Multilingual Dataset for Causal Commonsense Reasoning},
  author={Edoardo M. Ponti, Goran Glava\v{s}, Olga Majewska, Qianchu Liu, Ivan Vuli\'{c} and Anna Korhonen},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020},
  url={https://ducdauge.github.io/files/xcopa.pdf}
}
"""


class XCopa(Copa):
    VERSION = 0
    DATASET_PATH = "xcopa"
    DATASET_NAME = None
    CAUSE = "because"
    EFFECT = "therefore"

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def doc_to_text(self, doc):
        # Drop the period
        connector = {
            "cause": self.CAUSE,
            "effect": self.EFFECT,
        }[doc["question"]]
        return doc["premise"].strip()[:-1] + f" {connector}"


class XCopaEt(XCopa):
    DATASET_NAME = "et"
    CAUSE = "sest"
    EFFECT = "seetõttu"


class XCopaHt(XCopa):
    DATASET_NAME = "ht"
    CAUSE = "poukisa"
    EFFECT = "donk sa"


class XCopaIt(XCopa):
    DATASET_NAME = "it"
    CAUSE = "perché"
    EFFECT = "quindi"


class XCopaId(XCopa):
    DATASET_NAME = "id"
    CAUSE = "karena"
    EFFECT = "maka"


class XCopaQu(XCopa):
    DATASET_NAME = "qu"
    CAUSE = "imataq"
    EFFECT = "chaymi"


class XCopaSw(XCopa):
    DATASET_NAME = "sw"
    CAUSE = "kwa sababu"
    EFFECT = "kwa hiyo"


class XCopaZh(XCopa):
    DATASET_NAME = "zh"
    CAUSE = "因为"
    EFFECT = "所以"


class XCopaTa(XCopa):
    DATASET_NAME = "ta"
    CAUSE = "காரணமாக"
    EFFECT = "எனவே"


class XCopaTh(XCopa):
    DATASET_NAME = "th"
    CAUSE = "เพราะ"
    EFFECT = "ดังนั้น"


class XCopaTr(XCopa):
    DATASET_NAME = "tr"
    CAUSE = "çünkü"
    EFFECT = "bu yüzden"


class XCopaVi(XCopa):
    DATASET_NAME = "vi"
    CAUSE = "bởi vì"
    EFFECT = "vì vậy"


LANGS = ["et", "ht", "it", "id", "qu", "sw", "zh", "ta", "th", "tr", "vi"]

LANG_CLASSES = [
    XCopaEt,
    XCopaHt,
    XCopaIt,
    XCopaId,
    XCopaQu,
    XCopaSw,
    XCopaZh,
    XCopaTa,
    XCopaTh,
    XCopaTr,
    XCopaVi,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xcopa_{lang}"] = lang_class
    return tasks
