from squad import SQuAD2
from math import exp
from functools import partial
import datasets


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)
    return _squad_metric(predictions=predictions, references=references)[key]


class XQuADBase(SQuAD2):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation, (logprob_unanswerable, _) = results

        no_answer_probability = exp(logprob_unanswerable)

        predictions = {
            'id': doc['id'],
            'prediction_text': continuation,
            'no_answer_probability': no_answer_probability,
        }

        references = {
            'id': doc['id'],
            'answers': doc['answers'],
        }

        return {
            'exact': (predictions, references),  # Exact match (the normalized answer exactly match the gold answer)
            'f1': (predictions, references),  # The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            'exact': partial(_squad_agg, 'exact'),  # Exact match (the normalized answer exactly match the gold answer)
            'f1': partial(_squad_agg, 'f1'),  # The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            'exact': True,  # Exact match (the normalized answer exactly match the gold answer)
            'f1': True,  # The F-score of predicted tokens versus the gold answer
        }


class XQuADAr(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.ar'


class XQuADDe(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.de'


class XQuADZh(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.zh'


class XQuADVi(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.vi'


class XQuADEn(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.en'


class XQuADEs(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.es'


class XQuADHi(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.hi'


class XQuADEl(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.el'


class XQuADTh(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.th'


class XQuADTr(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.tr'


class XQuADRu(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.ru'


class XQuADRo(XQuADBase):
    VERSION = 0
    DATASET_PATH = "xquad"
    DATASET_NAME = 'xquad.ro'
