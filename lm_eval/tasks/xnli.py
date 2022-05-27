"""
XNLI: Evaluating Cross-lingual Sentence Representations
https://arxiv.org/abs/1809.05053

@InProceedings{conneau2018xnli,
  author = "Conneau, Alexis
        and Rinott, Ruty
        and Lample, Guillaume
        and Williams, Adina
        and Bowman, Samuel R.
        and Schwenk, Holger
        and Stoyanov, Veselin",
  title = "XNLI: Evaluating Cross-lingual Sentence Representations",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods
               in Natural Language Processing",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  location = "Brussels, Belgium",
}
"""


import numpy as np
from lm_eval.base import rf
from ..metrics import mean
from .common import HFTask

LANGS = [
    "ar",
    "bg",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "hi",
    "ru",
    "sw",
    "th",
    "tr",
    "ur",
    "vi",
    "zh",
]


class XNLIBase(HFTask):
    VERSION = 0
    DATASET_PATH = "xnli"
    DATASET_NAME = None

    QUESTION = ""
    ANSWER = ""
    TRUE = ""
    FALSE = ""
    NEITHER = ""
    OPTIONS = ""

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def doc_to_text(self, doc):
        return doc["premise"] + ", right? [MASK], " + doc["hypothesis"]

    def doc_to_target(self, doc):
        # True = entailment
        # False = contradiction
        # Neither = neutral
        return " " + [self.TRUE, self.NEITHER, self.FALSE][doc["label"]]

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        ll_true = rf.loglikelihood_rolling(ctx.replace("[MASK]", "Yes"))[0]
        ll_neither = rf.loglikelihood_rolling(ctx.replace("[MASK]", "Also"))[0]
        ll_false = rf.loglikelihood_rolling(ctx.replace("[MASK]", "No"))[0]
        return ll_true, ll_neither, ll_false

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        gold = doc["label"]
        pred = np.argmax(results)
        return {"acc": pred == gold}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {"acc": mean}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {"acc": True}


class XNLI_ar(XNLIBase):  # Arabic
    DATASET_NAME = "ar"

    QUESTION = ":سؤال"
    ANSWER = ":إِجابة"
    TRUE = "صحيح"
    FALSE = "كاذب"
    NEITHER = "حيادي"
    OPTIONS = "صحيح أو كاذب أو حيادي؟"


class XNLI_bg(XNLIBase):  # Bulgarian
    DATASET_NAME = "bg"

    QUESTION = "Въпрос:"
    ANSWER = "Отговор:"
    TRUE = "Правда"
    FALSE = "Ложный"
    NEITHER = "Нейтральный"
    OPTIONS = "Правда, Ложный или Нейтральный?"


class XNLI_de(XNLIBase):  # German
    DATASET_NAME = "de"

    QUESTION = "Frage:"
    ANSWER = "Antwort:"
    TRUE = "Stimmt"
    FALSE = "Falsch"
    NEITHER = "Neutral"
    OPTIONS = "Stimmt, Falsch oder Neutral?"


class XNLI_el(XNLIBase):  # Greek
    DATASET_NAME = "el"

    QUESTION = "Ερώτηση:"
    ANSWER = "Απάντηση:"
    TRUE = "Σωστό"
    FALSE = "Λάθος"
    NEITHER = "Ουδέτερο"
    OPTIONS = "Σωστό, Λάθος ή Ουδέτερο?"


class XNLI_en(XNLIBase):  # English
    DATASET_NAME = "en"

    QUESTION = "Question:"
    ANSWER = "Answer:"
    TRUE = "True"
    FALSE = "False"
    NEITHER = "Neither"
    OPTIONS = "True, False or Neither?"


class XNLI_es(XNLIBase):  # Spanish
    DATASET_NAME = "es"

    QUESTION = "Pregunta:"
    ANSWER = "Respuesta:"
    TRUE = "Verdad"
    FALSE = "Falsa"
    NEITHER = "Ninguno"
    OPTIONS = "Verdad, Falsa o Ninguno?"


class XNLI_fr(XNLIBase):  # French
    DATASET_NAME = "fr"

    QUESTION = "Question:"
    ANSWER = "Réponse:"
    TRUE = "Vrai"
    FALSE = "Faux"
    NEITHER = "Neutre"
    OPTIONS = "Vrai, Faux ou Neutre?"


class XNLI_hi(XNLIBase):  # Hindi
    DATASET_NAME = "hi"

    QUESTION = "प्रश्न:"
    ANSWER = "उत्तर:"
    TRUE = "सत्य"
    FALSE = "असत्य"
    NEITHER = "तटस्थ"
    OPTIONS = "सत्य या असत्य या तटस्थ?"


class XNLI_ru(XNLIBase):  # Russian
    DATASET_NAME = "ru"

    QUESTION = "Вопрос:"
    ANSWER = "Ответ:"
    TRUE = "Правда"
    FALSE = "Ложный"
    NEITHER = "Нейтральный"
    OPTIONS = "Правда, Ложный или Нейтральный?"


class XNLI_sw(XNLIBase):  # Swahili
    DATASET_NAME = "sw"

    QUESTION = "Swali:"
    ANSWER = "Jibu:"
    TRUE = "Kweli"
    FALSE = "Uongo"
    NEITHER = "Wala"
    OPTIONS = "Kweli, Uongo au Wala?"


class XNLI_th(XNLIBase):  # Thai
    DATASET_NAME = "th"

    QUESTION = "คำถาม:"
    ANSWER = "คำตอบ:"
    TRUE = "จริง"
    FALSE = "เท็จ"
    NEITHER = "เป็นกลาง"
    OPTIONS = "จริงหรือเท็จหรือเป็นกลาง?"


class XNLI_tr(XNLIBase):  # Turkish
    DATASET_NAME = "tr"

    QUESTION = "Soru:"
    ANSWER = "Cevap:"
    TRUE = "Doğru"
    FALSE = "Yanlış"
    NEITHER = "Nötr"
    OPTIONS = "Doğru, Yanlış veya Nötr?"


class XNLI_ur(XNLIBase):  # Urdu
    DATASET_NAME = "ur"

    QUESTION = ":سوال"
    ANSWER = ":جواب"
    TRUE = "صحیح"
    FALSE = "غلط"
    NEITHER = "غیر جانبدار"
    OPTIONS = "صحیح یا غلط یا غیر جانبدار؟"


class XNLI_vi(XNLIBase):  # Vietnamese
    DATASET_NAME = "vi"

    QUESTION = "Câu hỏi:"
    ANSWER = "Câu trả lời:"
    TRUE = "Đúng"
    FALSE = "Sai"
    NEITHER = "Trung lập"
    OPTIONS = "Đúng, Sai hay Trung lập?"


class XNLI_zh(XNLIBase):  # Chinese
    DATASET_NAME = "zh"

    QUESTION = "问题:"
    ANSWER = "回答:"
    TRUE = "对"
    FALSE = "错"
    NEITHER = "中立"
    OPTIONS = "对、错、还是中立?"


LANG_CLASSES = [
    XNLI_ar,
    XNLI_bg,
    XNLI_de,
    XNLI_el,
    XNLI_en,
    XNLI_es,
    XNLI_fr,
    XNLI_hi,
    XNLI_ru,
    XNLI_sw,
    XNLI_th,
    XNLI_tr,
    XNLI_ur,
    XNLI_vi,
    XNLI_zh,
]


def construct_tasks():
    tasks = {}
    for lang, lang_class in zip(LANGS, LANG_CLASSES):
        tasks[f"xnli_{lang}"] = lang_class
    return tasks
