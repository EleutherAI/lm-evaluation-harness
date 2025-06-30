import re
from abc import abstractmethod
from functools import reduce

import numpy as np
import transformers.data.metrics.squad_metrics as squad_metrics
from datasets import Dataset
from evaluate import load
from transformers import AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import mean
from lm_eval.api.task import ConfigurableTask


_CITATION = """
@inproceedings{shaham-etal-2022-scrolls,
    title = "{SCROLLS}: Standardized {C}ompa{R}ison Over Long Language Sequences",
    author = "Shaham, Uri  and
      Segal, Elad  and
      Ivgi, Maor  and
      Efrat, Avia  and
      Yoran, Ori  and
      Haviv, Adi  and
      Gupta, Ankit  and
      Xiong, Wenhan  and
      Geva, Mor  and
      Berant, Jonathan  and
      Levy, Omer",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.823",
    pages = "12007--12021"
}
"""

# SCROLLS is formualted as a sequence-to-sequence task.
# To allow for evaluation of causal models, we'll
# reformualte these with appropriate prompts


def _download_metric():
    import os
    import shutil

    from huggingface_hub import hf_hub_download

    scrolls_metric_path = hf_hub_download(
        repo_id="tau/scrolls",
        repo_type="dataset",
        filename="metrics/scrolls.py",
        revision="refs/pr/5",
    )
    updated_scrolls_metric_path = (
        os.path.dirname(scrolls_metric_path)
        + os.path.basename(scrolls_metric_path).replace(".", "_")
        + ".py"
    )
    shutil.copy(scrolls_metric_path, updated_scrolls_metric_path)
    return updated_scrolls_metric_path


def _process_doc_prepended_question(doc):
    # "When a query is given in addition to the raw text (as
    # in QMSum, Qasper, NarrativeQA, QuALITY, and ContractNLI),
    # we prepend it to the text, using two newlines as a natural separator"
    input = doc["input"]
    split = input.find("\n\n")
    return {
        "id": doc["id"],
        "pid": doc["pid"],
        "input": input,
        "outputs": doc["outputs"],
        "question": input[0:split],
        "text": input[split + 2 :],
    }


def _drop_duplicates_in_input(untokenized_dataset):
    # from scrolls/evaluator/dataset_evaluator.py

    indices_to_keep = []
    id_to_idx = {}
    outputs = []
    for i, (id_, output) in enumerate(
        zip(untokenized_dataset["id"], untokenized_dataset["output"])
    ):
        if id_ in id_to_idx:
            outputs[id_to_idx[id_]].append(output)
            continue
        indices_to_keep.append(i)
        id_to_idx[id_] = len(outputs)
        outputs.append([output])
    untokenized_dataset = untokenized_dataset.select(indices_to_keep).flatten_indices()
    untokenized_dataset = untokenized_dataset.remove_columns("output")
    untokenized_dataset = untokenized_dataset.add_column("outputs", outputs)
    return untokenized_dataset


def _num_cpu_cores():
    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python/55423170#55423170
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except ImportError:
        import os

        return len(os.sched_getaffinity(0))


class _SCROLLSTask(ConfigurableTask):
    VERSION = 2
    DATASET_PATH = "tau/scrolls"
    DATASET_NAME = None
    PRUNE_TOKENIZERS = None
    PRUNE_MAX_TOKENS = None
    PRUNE_NUM_PROC = None

    def __init__(self, config=None):
        super().__init__(config={"metadata": {"version": self.VERSION}})
        if self.DATASET_NAME is not None:
            self.metric = load(_download_metric(), config_name=self.DATASET_NAME)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        processed_docs = list(map(self._process_doc, self.dataset["train"]))

        # Flatten the list of lists since _process_doc returns a list of one element.
        processed_docs = [item for sublist in processed_docs for item in sublist]
        processed_dict = {
            key: [d[key] for d in processed_docs] for key in processed_docs[0]
        }

        return Dataset.from_dict(processed_dict)

    def validation_docs(self):
        processed_docs = list(map(self._process_doc, self.dataset["validation"]))

        # Flatten the list of lists since _process_doc returns a list of one element.
        processed_docs = [item for sublist in processed_docs for item in sublist]
        processed_dict = {
            key: [d[key] for d in processed_docs] for key in processed_docs[0]
        }

        return Dataset.from_dict(processed_dict)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["input"]

    def download(self, *args, **kwargs):
        super().download(*args, **kwargs)
        del self.dataset["test"]
        for split in self.dataset:
            self.dataset[split] = _drop_duplicates_in_input(self.dataset[split])
        if self.PRUNE_TOKENIZERS is not None:
            self.prune()

    def _get_prune_text(self, sample):
        return self.doc_to_text(self._process_doc(sample)[0])

    def prune(self):
        """Create a pruned version of a SCROLLS task dataset containing only inputs
        that are less than `max_tokens` when tokenized by each tokenizer
        """

        tokenizers = [
            AutoTokenizer.from_pretrained(tokenizer)
            for tokenizer in self.PRUNE_TOKENIZERS
        ]
        cache = {}

        def _filter(sample):
            text = self._get_prune_text(sample)
            cached = cache.get(text, None)
            if cached is None:
                for tokenizer in tokenizers:
                    if len(tokenizer(text).input_ids) > self.PRUNE_MAX_TOKENS:
                        cache[text] = False
                        return False
                cache[text] = True
                return True
            else:
                return cached

        self.dataset = self.dataset.filter(_filter, num_proc=self.PRUNE_NUM_PROC)

    def doc_to_target(self, doc):
        return " " + ", ".join(doc["outputs"])

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"

    def higher_is_better(self):
        return {x: True for x in self._scrolls_metrics().keys()}

    @abstractmethod
    def _scrolls_metrics(self):
        pass

    def _make_compute_metrics(self, value):
        def compute_metrics(samples):
            predictions, references = zip(*samples)  # unzip, if you will
            computed = self.metric.compute(
                predictions=predictions, references=references
            )
            return computed[value]

        return compute_metrics

    def aggregation(self):
        return {
            key: self._make_compute_metrics(value)
            for key, value in self._scrolls_metrics().items()
        }


class _SCROLLSMultipleChoiceTask(_SCROLLSTask):
    def __post_init__(self):
        self.metric = None

    def _scrolls_metrics(self):
        return None

    def aggregation(self):
        return {"em": mean, "acc": mean, "acc_norm": mean}

    def higher_is_better(self):
        return {"em": True, "acc": True, "acc_norm": True}

    def process_results(self, doc, results):
        gold = doc["gold"]

        lls, _ = zip(*results)
        acc = 1.0 if np.argmax(lls) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(lls / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "em": acc_norm * 100.0,
        }

    def construct_requests(self, doc, ctx, **kwargs):
        apply_chat_template = kwargs.pop("apply_chat_template", False)
        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " {}".format(choice))
                if not apply_chat_template
                else (ctx, "{}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]
        return request_list


class _SCROLLSSummaryTask(_SCROLLSTask):
    def _process_doc(self, doc):
        return [doc]

    def _scrolls_metrics(self):
        return {
            "rouge1": "rouge/rouge1",
            "rouge2": "rouge/rouge2",
            "rougeL": "rouge/rougeL",
        }

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["outputs"]),
            "rouge2": (results[0], doc["outputs"]),
            "rougeL": (results[0], doc["outputs"]),
        }

    def construct_requests(self, doc, ctx, **kwargs):
        kwargs.pop("apply_chat_template", False)
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            **kwargs,
        )

    def doc_to_text(self, doc):
        return f"{doc['input']}\n\nQuestion: What is a summary of the preceding text?\nAnswer:"


class Qasper(_SCROLLSTask):
    """A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers
    https://arxiv.org/abs/2105.03011
    """

    DATASET_NAME = "qasper"

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)
        doc["is_yes_no"] = reduce(
            lambda prev, cur: prev
            and squad_metrics.normalize_answer(cur) in ["yes", "no"],
            doc["outputs"],
            True,
        )
        return [doc]

    def _scrolls_metrics(self):
        return {"f1": "f1"}

    def process_results(self, doc, results):
        if doc["is_yes_no"]:
            prediction = " yes" if results[0] > results[1] else " no"
        elif len(results[0].strip()) == 0:
            prediction = "Unanswerable"
        else:
            prediction = results[0]
        return {"f1": (prediction, doc["outputs"])}

    def construct_requests(self, doc, ctx, **kwargs):
        apply_chat_template = kwargs.pop("apply_chat_template", False)
        if doc["is_yes_no"]:
            return [
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " yes")
                    if not apply_chat_template
                    else (ctx, "yes"),
                    idx=0,
                    **kwargs,
                ),
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " no") if not apply_chat_template else (ctx, "no"),
                    idx=1,
                    **kwargs,
                ),
            ]
        else:
            return Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": ["\n"]}),
                idx=0,
                **kwargs,
            )


class QuALITY(_SCROLLSMultipleChoiceTask):
    """QuALITY: Question Answering with Long Input Texts, Yes!
    https://arxiv.org/abs/2112.08608
    """

    DATASET_NAME = "quality"
    _multiple_choice_pattern = re.compile(r" *\([A-D]\) *")

    @staticmethod
    def _normalize_answer(text):
        return " ".join(text.split()).strip()

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)

        split = doc["text"].find("\n\n", doc["text"].find("(D)"))
        choices_text = doc["text"][:split]

        doc["text"] = doc["text"][split:].strip()
        doc["choices"] = [
            QuALITY._normalize_answer(choice)
            for choice in re.split(QuALITY._multiple_choice_pattern, choices_text)[1:]
        ]
        doc["gold"] = doc["choices"].index(QuALITY._normalize_answer(doc["outputs"][0]))

        return [doc]


class NarrativeQA(_SCROLLSTask):
    """The NarrativeQA Reading Comprehension Challenge
    https://arxiv.org/abs/1712.07040
    """

    DATASET_NAME = "narrative_qa"

    def _process_doc(self, doc):
        return [_process_doc_prepended_question(doc)]

    def _scrolls_metrics(self):
        return {"f1": "f1"}

    def _get_prune_text(self, doc):
        # pruning narrativeqa takes forever -- let's cheat a bit
        # and just cache on the text, not the question, since
        # the dataset is different questions about the same large
        # documents
        return self._process_doc(doc)[0]["text"]

    def process_results(self, doc, results):
        return {"f1": (results[0], doc["outputs"])}

    def construct_requests(self, doc, ctx, **kwargs):
        kwargs.pop("apply_chat_template", False)
        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {"until": ["\n"]}),
            idx=0,
            **kwargs,
        )


class ContractNLI(_SCROLLSMultipleChoiceTask):
    """ContractNLI: A Dataset for Document-level Natural Language Inference for Contracts
    https://arxiv.org/abs/1712.07040
    """

    DATASET_NAME = "contract_nli"
    CHOICES = ["Not mentioned", "Entailment", "Contradiction"]

    def _process_doc(self, doc):
        doc = _process_doc_prepended_question(doc)
        doc["choices"] = ContractNLI.CHOICES
        doc["gold"] = ContractNLI.CHOICES.index(doc["outputs"][0])
        return [doc]

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nHypothesis: {doc['question']}\nConclusion:"


class GovReport(_SCROLLSSummaryTask):
    """Efficient Attentions for Long Document Summarization
    https://arxiv.org/abs/2104.02112

    Note: The average length of the reference summaries is ~3,000
    characters, or ~600 tokens as tokenized by GPT-NeoX. For causal models,
    it is recommended to set `max_gen_toks` sufficiently large (e.g. 1024)
    to allow a full summary to be generated.
    """

    DATASET_NAME = "gov_report"


class SummScreenFD(_SCROLLSSummaryTask):
    """SummScreen: A Dataset for Abstractive Screenplay Summarization
    https://arxiv.org/abs/2104.07091
    """

    DATASET_NAME = "summ_screen_fd"


class QMSum(_SCROLLSSummaryTask):
    """QMSum: A New Benchmark for Query-based Multi-domain
    Meeting Summarization

    https://arxiv.org/abs/2104.05938
    """

    DATASET_NAME = "qmsum"

    def _process_doc(self, doc):
        return [_process_doc_prepended_question(doc)]

    def doc_to_text(self, doc):
        return f"{doc['text']}\n\nQuestion: {doc['question']}\nAnswer:"
