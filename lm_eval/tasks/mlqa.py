from . common import HFTask
from math import exp
from functools import partial
import sys, unicodedata, string
import re
from collections import Counter
from lm_eval.base import rf

PUNCT = {chr(i) for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')}.union(
    string.punctuation)
WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar']
MIXED_SEGMENTATION_LANGS = ['zh']
ALL_MLQA_SETS_TRANSLATE = ['mlqa-translate-train.ar', 'mlqa-translate-train.de', 'mlqa-translate-train.vi',
                           'mlqa-translate-train.zh', 'mlqa-translate-train.es', 'mlqa-translate-train.hi',
                           'mlqa-translate-test.ar', 'mlqa-translate-test.de', 'mlqa-translate-test.vi',
                           'mlqa-translate-test.zh',
                           'mlqa-translate-test.es', 'mlqa-translate-test.hi']

ALL_MLQA_SETS_QA = ['mlqa.ar.ar', 'mlqa.ar.de', 'mlqa.ar.vi',
                    'mlqa.ar.zh', 'mlqa.ar.en', 'mlqa.ar.es', 'mlqa.ar.hi', 'mlqa.de.ar', 'mlqa.de.de', 'mlqa.de.vi',
                    'mlqa.de.zh', 'mlqa.de.en', 'mlqa.de.es', 'mlqa.de.hi', 'mlqa.vi.ar', 'mlqa.vi.de', 'mlqa.vi.vi',
                    'mlqa.vi.zh', 'mlqa.vi.en', 'mlqa.vi.es', 'mlqa.vi.hi', 'mlqa.zh.ar', 'mlqa.zh.de', 'mlqa.zh.vi',
                    'mlqa.zh.zh', 'mlqa.zh.en', 'mlqa.zh.es', 'mlqa.zh.hi', 'mlqa.en.ar', 'mlqa.en.de', 'mlqa.en.vi',
                    'mlqa.en.zh', 'mlqa.en.en', 'mlqa.en.es', 'mlqa.en.hi', 'mlqa.es.ar', 'mlqa.es.de', 'mlqa.es.vi',
                    'mlqa.es.zh', 'mlqa.es.en', 'mlqa.es.es', 'mlqa.es.hi', 'mlqa.hi.ar', 'mlqa.hi.de', 'mlqa.hi.vi',
                    'mlqa.hi.zh', 'mlqa.hi.en', 'mlqa.hi.es', 'mlqa.hi.hi']

ALL_MLQA_SETS = ALL_MLQA_SETS_QA + ALL_MLQA_SETS_TRANSLATE

BG_Q_A_STRINGS = {
    "en": {"bg": "Background:", "q": "Question:", "a": "Answer:"},
    "ar": {"bg": ":معرفتي", "q": ":سؤال", "a": ":إجابه"},
    "de": {"bg": "Hintergrund:", "q": "Frage:", "a": "Antworten:"},
    "zh": {"bg": "背景:", "q": "問題:", "a": "回答:"},
    "vi": {"bg": "lý lịch:", "q": "câu hỏi:", "a": "câu trả lời:"},
    "es": {"bg": "antecedentes:", "q": "pregunta:", "a": "respuesta:"},
    "hi": {"bg": "Ιστορικό:", "q": "ερώτηση:", "a": "απάντηση:"},
}


def whitespace_tokenize(text):
    return text.split()


def mixed_segmentation(text):
    segs_out = []
    temp_str = ""
    for char in text:
        if re.search(r'[\u4e00-\u9fa5]', char) or char in PUNCT:
            if temp_str != "":
                ss = whitespace_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    if temp_str != "":
        ss = whitespace_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


def normalize_answer(s, lang):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text, lang):
        if lang == 'en':
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        elif lang == 'es':
            return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        elif lang == 'hi':
            return text  # Hindi does not have formal articles
        elif lang == 'vi':
            return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        elif lang == 'de':
            return re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        elif lang == 'ar':
            return re.sub('\sال^|ال', ' ', text)
        elif lang == 'zh':
            return text  # Chinese does not have formal articles
        else:
            raise Exception('Unknown Language {}'.format(lang))

    def white_space_fix(text, lang):
        if lang in WHITESPACE_LANGS:
            tokens = whitespace_tokenize(text)
        elif lang in MIXED_SEGMENTATION_LANGS:
            tokens = mixed_segmentation(text)
        else:
            raise Exception('Unknown Language {}'.format(lang))
        return ' '.join([t for t in tokens if t.strip() != ''])

    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in PUNCT)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)), lang), lang)


def f1_score(prediction, ground_truth, lang):
    prediction_tokens = normalize_answer(prediction, lang).split()
    ground_truth_tokens = normalize_answer(ground_truth, lang).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth, lang):
    return normalize_answer(prediction, lang) == normalize_answer(ground_truth, lang)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, lang):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, lang)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, answer_lang):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths, answer_lang)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths, answer_lang)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}


def mlqa_metric(predictions, references, answer_lang):
    pred_dict = {prediction['id']: prediction['prediction_text'] for prediction in predictions}
    dataset = [
        {
            "paragraphs": [
                {
                    "qas": [
                        {
                            "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                            "id": ref["id"],
                        }
                        for ref in references
                    ]
                }
            ]
        }
    ]
    return evaluate(dataset, pred_dict, answer_lang)


def mlqa_agg(items, key,  answer_lang):
    predictions, references = zip(*items)
    return mlqa_metric(predictions=predictions, references=references, answer_lang=answer_lang)[key]


class MLQABase(HFTask):

    def __init__(self, dataset_name=None, question_lang=None, answer_lang=None, background='Background:',
                 question='Question:', answer="Answer:"):
        self.VERSION = 0
        self.DATASET_PATH = "mlqa"
        self.DATASET_NAME = dataset_name
        self.QUESTION_LANG = question_lang
        self.ANSWER_LANG = answer_lang
        self.BACKGROUND = background
        self.QUESTION = question
        self.ANSWER = answer
        super().__init__()

    def fewshot_description(self):
        # TODO: figure out description
        return ""

    def doc_to_text(self, doc):
        text = self.BACKGROUND + '\n\n' + doc['context'] + '\n\n' + self.QUESTION + doc['question'] + '\n\n' + \
               self.ANSWER
        return text

    def doc_to_target(self, doc):
        answer_list = doc['answers']['text']
        if len(answer_list) > 0:
            answer = answer_list[0]
        else:
            answer = 'unanswerable'
        return " " + answer

    def construct_requests(self, doc, ctx):
        """ Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        continuation = rf.greedy_until(ctx, ['\n'])
        return continuation

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        continuation = results
        
        if isinstance(continuation, list):
            continuation = continuation[0]

        predictions = {
            'id': doc['id'],
            'prediction_text': continuation,
        }

        references = {
            'id': doc['id'],
            'answers': doc['answers'],
        }

        return {
            'exact_match': (predictions, references),  # Exact match (the normalized answer exactly match the gold answer)
            'f1': (predictions, references),  # The F-score of predicted tokens versus the gold answer
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            'exact_match': partial(mlqa_agg, key='exact_match', answer_lang=self.ANSWER_LANG),  # Exact match (the normalized
            # answer exactly match the gold answer)
            'f1': partial(mlqa_agg, key='f1', answer_lang=self.ANSWER_LANG),  # The F-score of predicted tokens
            # versus the gold answer
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            'exact_match': True,  # Exact match (the normalized answer exactly match the gold answer)
            'f1': True,  # The F-score of predicted tokens versus the gold answer
        }


def construct_tasks():
    tasks = {}
    for name in ALL_MLQA_SETS_QA:
        parts = name.split('.')
        question_lang, answer_lang = parts[1], parts[2]
        bg_qa_string = BG_Q_A_STRINGS[answer_lang]
        tasks[name] = partial(MLQABase, dataset_name=name, question_lang=question_lang, answer_lang=answer_lang,
                              background=bg_qa_string['bg'], question=bg_qa_string['q'], answer=bg_qa_string['a'])
    return tasks

