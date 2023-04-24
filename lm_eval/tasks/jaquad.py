# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.

TODO: Write a Short Description of the task.

Homepage: TODO: Add the URL to the task's Homepage here.


Japanese Question Answering Dataset (JaQuAD), released in 2022, is a human-annotated dataset created for Japanese Machine Reading Comprehension. JaQuAD is developed to provide a SQuAD-like QA dataset in Japanese. JaQuAD contains 39,696 question-answer pairs. Questions and answers are manually curated by human annotators. Contexts are collected from Japanese Wikipedia articles. Fine-tuning BERT-Japanese on JaQuAD achieves 78.92% for an F1 score and 63.38% for an exact match.
"""
from functools import partial
import datasets
from lm_eval.base import rf, Task


_CITATION = """
@misc{so2022jaquad,
      title={{JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension}},
      author={ByungHoon So and Kyuhong Byun and Kyungwon Kang and Seongjin Cho},
      year={2022},
      eprint={2202.01764},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

"""


def _squad_metric(predictions, references):
    squad_metric = datasets.load_metric("squad")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, item):
    predictions, references = zip(*item)
    return _squad_metric(predictions=predictions, references=references)[key]


class Jaquad(Task):
    VERSION = 0
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "SkelterLabsInc/JaQuAD"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None  # "0.1.0"

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                # TODO: Return the training document generator from `self.dataset`.
                # If you need to process the data, `map` over the documents with
                # the custom processing function, `self._process_doc`. E.g.
                # `map(self._process_doc, self.dataset["validation"])`
                # In most case you can leave this as is unless the dataset split is
                # named differently than the default `"train"`.
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            # TODO: Return the validation document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["validation"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"validation"`.
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            # TODO: Return the test document generator from `self.dataset`.
            # If you need to process the data, `map` over the documents with the
            # custom processing function, `self._process_doc`. E.g.
            # `map(self._process_doc, self.dataset["test"])`
            # In most case you can leave this as is unless the dataset split is
            # named differently than the default `"test"`.
            return self.dataset["test"]

    def _process_doc(self, doc):
        # TODO: Process (detokenize, strip, replace etc.) each individual `doc`
        # with this function. You can map this across the docs in each available
        # dataset split. See the TODOs in `train_docs`, `validation_docs`, and
        # `test_docs` for snippets.
        # NOTE: DELETE THIS FUNCTION IF UNUSED.
        return doc

    def doc_to_text(self, doc):
        return 'コンテキスト: ' + doc['context'] + '\n\n' + '質問: ' + doc['question'] + '\n\n' + '回答:'

    def doc_to_target(self, doc):
        target_list = doc['answers']['text']
        target = target_list[0]
        return " " + target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
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
        predictions = {
            "id": doc["id"],
            "prediction_text": continuation,
        }
        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }
        return {
            "exact_match": (predictions, references),
            "f1": (predictions, references),
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            'exact_match': partial(_squad_agg, 'exact_match'), # Exact match (the normalized answer exactly match the gold answer)
            'f1': partial(_squad_agg, 'f1'), #  The F-score of predicted tokens versus the gold answer
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            'exact_match': True,
            'f1': True,
        }
