"""
Common Sense Beyond English: Evaluating and Improving Multilingual Language Models for Commonsense Reasoning
https://aclanthology.org/2021.acl-long.102.pdf

X-CSR consists of two different datasets: X-CSQA and X-CODAH
X-CommonsenseQA (X-CSQA) is a multiple-choice QA task targeting general commonsense knowledge,
however, it only has English version. X-CODAH dataset is a scene completion task with options,
which shares a similar format to CSQA. Those two datasets are used to evaluate multi-lingual
language models (ML-LMs) for commonsense reasoning in a cross-lingual zero-shot transfer setting.
The total 16 languages for X-CSR: {en, zh, de, es, fr, it, jap, nl, pl, pt, ru, ar, vi, hi, sw, ur}.

Homepage: https://inklab.usc.edu/XCSR/xcsr_datasets
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@inproceedings{lin-etal-2021-xcsr,
    title = "Common Sense Beyond English: Evaluating and Improving Multilingual Language Models for Commonsense Reasoning",
    author = "Lin, Bill Yuchen and Lee, Seyeon and Qiao, Xiaoyang and Ren, Xiang",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL-IJCNLP 2021)",
    year = "2021",
    note={to appear}
}
"""


class XCSQABase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "xcsr"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["question"]["stem"],
            "choices": doc["question"]["choices"]["text"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class XCODAHBase(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "xcsr"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def _process_doc(self, doc):
        out_doc = {
            "query": doc["question"]["stem"],
            "choices": doc["question"]["choices"]["text"],
            "gold": ["A", "B", "C", "D"].index(doc["answerKey"].strip()),
        }
        return out_doc

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]


class XCSQA_ar(XCSQABase):  # Arabic
    DATASET_NAME = "X-CSQA-ar"


class XCSQA_de(XCSQABase):  # German
    DATASET_NAME = "X-CSQA-de"


class XCSQA_en(XCSQABase):  # English
    DATASET_NAME = "X-CSQA-en"


class XCSQA_es(XCSQABase):  # Spanish
    DATASET_NAME = "X-CSQA-es"


class XCSQA_fr(XCSQABase):  # French
    DATASET_NAME = "X-CSQA-fr"


class XCSQA_hi(XCSQABase):  # Hindi
    DATASET_NAME = "X-CSQA-hi"


class XCSQA_it(XCSQABase):  # Italian
    DATASET_NAME = "X-CSQA-it"


class XCSQA_jap(XCSQABase):  # Japanese
    DATASET_NAME = "X-CSQA-jap"


class XCSQA_nl(XCSQABase):  # Dutch
    DATASET_NAME = "X-CSQA-nl"


class XCSQA_pl(XCSQABase):  # Polish
    DATASET_NAME = "X-CSQA-pl"


class XCSQA_pt(XCSQABase):  # Portuguese
    DATASET_NAME = "X-CSQA-pt"


class XCSQA_ru(XCSQABase):  # Russian
    DATASET_NAME = "X-CSQA-ru"


class XCSQA_sw(XCSQABase):  # Swahili
    DATASET_NAME = "X-CSQA-sw"


class XCSQA_ur(XCSQABase):  # Urdu
    DATASET_NAME = "X-CSQA-ur"


class XCSQA_vi(XCSQABase):  # Vietnamese
    DATASET_NAME = "X-CSQA-vi"


class XCSQA_zh(XCSQABase):  # Chinese
    DATASET_NAME = "X-CSQA-zh"


class XCODAH_ar(XCODAHBase):  # Arabic
    DATASET_NAME = "X-CODAH-ar"


class XCODAH_de(XCODAHBase):  # German
    DATASET_NAME = "X-CODAH-de"


class XCODAH_en(XCODAHBase):  # English
    DATASET_NAME = "X-CODAH-en"


class XCODAH_es(XCODAHBase):  # Spanish
    DATASET_NAME = "X-CODAH-es"


class XCODAH_fr(XCODAHBase):  # French
    DATASET_NAME = "X-CODAH-fr"


class XCODAH_hi(XCODAHBase):  # Hindi
    DATASET_NAME = "X-CODAH-hi"


class XCODAH_it(XCODAHBase):  # Italian
    DATASET_NAME = "X-CODAH-it"


class XCODAH_jap(XCODAHBase):  # Japanese
    DATASET_NAME = "X-CODAH-jap"


class XCODAH_nl(XCODAHBase):  # Dutch
    DATASET_NAME = "X-CODAH-nl"


class XCODAH_pl(XCODAHBase):  # Polish
    DATASET_NAME = "X-CODAH-pl"


class XCODAH_pt(XCODAHBase):  # Portuguese
    DATASET_NAME = "X-CODAH-pt"


class XCODAH_ru(XCODAHBase):  # Russian
    DATASET_NAME = "X-CODAH-ru"


class XCODAH_sw(XCODAHBase):  # Swahili
    DATASET_NAME = "X-CODAH-sw"


class XCODAH_ur(XCODAHBase):  # Urdu
    DATASET_NAME = "X-CODAH-ur"


class XCODAH_vi(XCODAHBase):  # Vietnamese
    DATASET_NAME = "X-CODAH-vi"


class XCODAH_zh(XCODAHBase):  # Chinese
    DATASET_NAME = "X-CODAH-zh"


LANGS_XCSQA = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "jap",
    "nl",
    "pl",
    "pt",
    "ru",
    "sw",
    "ur",
    "vi",
    "zh",
]

LANG_CLASSES_XCSQA = [
    XCSQA_ar,
    XCSQA_de,
    XCSQA_en,
    XCSQA_es,
    XCSQA_fr,
    XCSQA_hi,
    XCSQA_it,
    XCSQA_jap,
    XCSQA_nl,
    XCSQA_pl,
    XCSQA_pt,
    XCSQA_ru,
    XCSQA_sw,
    XCSQA_ur,
    XCSQA_vi,
    XCSQA_zh,
]

LANGS_XCODAH = [
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "it",
    "jap",
    "nl",
    "pl",
    "pt",
    "ru",
    "sw",
    "ur",
    "vi",
    "zh",
]
LANG_CLASSES_XCODAH = [
    XCODAH_ar,
    XCODAH_de,
    XCODAH_en,
    XCODAH_es,
    XCODAH_fr,
    XCODAH_hi,
    XCODAH_it,
    XCODAH_jap,
    XCODAH_nl,
    XCODAH_pl,
    XCODAH_pt,
    XCODAH_ru,
    XCODAH_sw,
    XCODAH_ur,
    XCODAH_vi,
    XCODAH_zh,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS_XCSQA, LANG_CLASSES_XCSQA):
        tasks[f"xcsqa_{lang}"] = lang_class
    for lang, lang_class in zip(LANGS_XCODAH, LANG_CLASSES_XCODAH):
        tasks[f"xcodah_{lang}"] = lang_class
    return tasks
