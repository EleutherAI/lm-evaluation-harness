"""
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
https://aclanthology.org/attachments/N19-1246.Supplementary.pdf

DROP is a QA dataset which tests comprehensive understanding of paragraphs. In
this crowdsourced, adversarially-created, 96k question-answering benchmark, a
system must resolve multiple references in a question, map them onto a paragraph,
and perform discrete operations over them (such as addition, counting, or sorting).

Homepage: https://allenai.org/data/drop

Acknowledgement: This implementation is based on the official evaluation for `DROP`:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
"""
import numpy as np
import re
import string
from scipy.optimize import linear_sum_assignment

from lm_eval.api.task import PromptSourceTask
from lm_eval.api.metric import mean


_CITATION = """
@misc{dua2019drop,
    title={DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs},
    author={Dheeru Dua and Yizhong Wang and Pradeep Dasigi and Gabriel Stanovsky and Sameer Singh and Matt Gardner},
    year={2019},
    eprint={1903.00161},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""


_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


class DROP(PromptSourceTask):
    VERSION = 1
    DATASET_PATH = "drop"  # inspect.getfile(lm_eval.datasets.drop.drop)
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        return self.dataset["train"]

    def validation_docs(self):
        return self.dataset["validation"]

    def process_results(self, doc, results):
        golds = self.doc_to_target(doc)
        pred = results[0].strip()
        preds = [pred]

        max_em = 0
        max_f1 = 0
        for gold_answer in golds:
            exact_match, f1_score = self.get_metrics(preds, gold_answer)
            if gold_answer[0].strip():
                max_em = max(max_em, exact_match)
                max_f1 = max(max_f1, f1_score)

        if self.save_examples:
            return {"em": max_em, "f1": max_f1}, {"pred": pred, "target": golds}
        return {"em": max_em, "f1": max_f1}

    def get_metrics(self, predicted, gold):
        """
        Takes a predicted answer and a gold answer (that are both either a string or a list of
        strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
        writing a script for evaluating objects in memory (say, the output of predictions during
        validation, or while training), this is the function you want to call, after using
        :func:`answer_json_to_strings` when reading the gold answer from the released data file.
        """
        predicted_bags = self._answer_to_bags(predicted)
        gold_bags = self._answer_to_bags(gold)

        if set(predicted_bags[0]) == set(gold_bags[0]) and len(
            predicted_bags[0]
        ) == len(gold_bags[0]):
            exact_match = 1.0
        else:
            exact_match = 0.0

        f1_per_bag = self._align_bags(predicted_bags[1], gold_bags[1])
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        return exact_match, f1

    def _answer_to_bags(self, answer):
        if isinstance(answer, (list, tuple)):
            raw_spans = answer
        else:
            raw_spans = [answer]
        normalized_spans = []
        token_bags = []
        for raw_span in raw_spans:
            normalized_span = self._normalize(raw_span)
            normalized_spans.append(normalized_span)
            token_bags.append(set(normalized_span.split()))
        return normalized_spans, token_bags

    def _align_bags(self, predicted, gold):
        """
        Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
        between them and gets maximum metric values over all the answers.
        """
        scores = np.zeros([len(gold), len(predicted)])
        for gold_index, gold_item in enumerate(gold):
            for pred_index, pred_item in enumerate(predicted):
                if self._match_numbers_if_present(gold_item, pred_item):
                    scores[gold_index, pred_index] = self._compute_f1(
                        pred_item, gold_item
                    )
        row_ind, col_ind = linear_sum_assignment(-scores)

        max_scores = np.zeros([max(len(gold), len(predicted))])
        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])
        return max_scores

    def _compute_f1(self, predicted_bag, gold_bag):
        intersection = len(gold_bag.intersection(predicted_bag))
        if not predicted_bag:
            precision = 1.0
        else:
            precision = intersection / float(len(predicted_bag))
        if not gold_bag:
            recall = 1.0
        else:
            recall = intersection / float(len(gold_bag))
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if not (precision == 0.0 and recall == 0.0)
            else 0.0
        )
        return f1

    def _match_numbers_if_present(self, gold_bag, predicted_bag):
        gold_numbers = set()
        predicted_numbers = set()
        for word in gold_bag:
            if self._is_number(word):
                gold_numbers.add(word)
        for word in predicted_bag:
            if self._is_number(word):
                predicted_numbers.add(word)
        if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
            return True
        return False

    def _is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _remove_articles(self, text):
        return _ARTICLES.sub(" ", text)

    def _white_space_fix(self, text):
        return " ".join(text.split())

    def _remove_punc(self, text):
        exclude = set(string.punctuation)
        if not self._is_number(text):
            return "".join(ch for ch in text if ch not in exclude)
        else:
            return text

    def _fix_number(self, text):
        return str(float(text)) if self._is_number(text) else text

    def _tokenize(self, text):
        return re.split(" |-", text)

    def _normalize(self, answer):
        tokens = [
            self._white_space_fix(
                self._remove_articles(
                    self._fix_number(self._remove_punc(token.lower()))
                )
            )
            for token in self._tokenize(answer)
        ]
        tokens = [token for token in tokens if token.strip()]
        normalized = " ".join(tokens).strip()
        return normalized

    def aggregation(self):
        return {"em": mean, "f1": mean}

    def higher_is_better(self):
        return {"em": True, "f1": True}
