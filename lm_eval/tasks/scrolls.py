"""
SCROLLS: Standardized CompaRison Over Long Language Sequences
https://arxiv.org/abs/2201.03533

SCROLLS is a suite of datasets that require synthesizing information over long texts.
The benchmark includes seven natural language tasks across multiple domains,
including summarization, question answering, and natural language inference.

Homepage: https://www.scrolls-benchmark.com/

Since SCROLLS tasks are generally longer than the maximum sequence length of many models,
it is possible to create "subset" tasks that contain only those samples whose tokenized length
is less than some pre-defined limit. For example, to create a subset of "Qasper" that would
be suitable for a model using the GPTNeoX tokenizer and a 4K maximium sequence length:

```
class QasperGPTNeoX4K(Qasper):
    PRUNE_TOKENIZERS = ["EleutherAI/pythia-410m-deduped"]
    PRUNE_MAX_TOKENS = 4096
    PRUNE_NUM_PROC = _num_cpu_cores() # optional, to speed up pruning of large datasets like NarrativeQA
```

`PRUNE_TOKENIZERS` can contain more than one tokenizer; this will include only samples that are
less than `PRUNE_MAX_TOKENS` for ALL of the tokenizers. This can be useful to comparing models
that use different tokenizers but the same maximum sequence length.

Once the subset task class has been defined in this file, it can be used by adding the class
to `lm_eval/tasks/__init__.py`.

NOTE: GovReport may need `max_gen_toks` set larger for causal models.
"""
from abc import abstractmethod
from datasets import load_metric
from transformers import AutoTokenizer
from lm_eval.base import rf, Task
from lm_eval.metrics import mean
from functools import reduce
import transformers.data.metrics.squad_metrics as squad_metrics
import numpy as np
import re

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
        repo_id="tau/scrolls", repo_type="dataset", filename="metrics/scrolls.py"
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


class _SCROLLSTask(Task):
    VERSION = 0
    DATASET_PATH = "tau/scrolls"
    DATASET_NAME = None
    PRUNE_TOKENIZERS = None
    PRUNE_MAX_TOKENS = None
    PRUNE_NUM_PROC = None

    def __init__(self, no_metric=False):
        super().__init__()
        self.metric = (
            load_metric(_download_metric(), config_name=self.DATASET_NAME)
            if not no_metric
            else None
        )

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def training_docs(self):
        for doc in self.dataset["train"]:
            yield from self._process_doc(doc)

    def validation_docs(self):
        for doc in self.dataset["validation"]:
            yield from self._process_doc(doc)

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["input"]

    def download(self, *args, **kwargs):
        super().download(*args, **kwargs)
        del self.dataset["test"]
        for split in self.dataset:
            self.dataset[split] = _drop_duplicates_in_input(self.dataset[split])
        if self.PRUNE_TOKENIZERS is not None and self.PRUNE_TOKENIZERS is not None:
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
    def __init__(self):
        super().__init__(no_metric=True)

    def _scrolls_metrics(self):
        return None

    def aggregation(self):
        return {"em": mean, "acc": mean, "acc_norm": mean}

    def higher_is_better(self):
        return {"em": True, "acc": True, "acc_norm": True}

    def process_results(self, doc, results):
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
            "em": acc_norm * 100.0,
        }

    def construct_requests(self, doc, ctx):
        lls = [
            rf.loglikelihood(ctx, " {}".format(choice))[0] for choice in doc["choices"]
        ]

        return lls


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

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {"until": ["\n"]})]

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

    def construct_requests(self, doc, ctx):
        if doc["is_yes_no"]:
            ll_yes, _ = rf.loglikelihood(ctx, " yes")
            ll_no, _ = rf.loglikelihood(ctx, " no")
            return [ll_yes, ll_no]
        else:
            return [rf.greedy_until(ctx, {"until": ["\n"]})]


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

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {"until": ["\n"]})]


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
    it is recommended to set `max_gen_toks` sufficently large (e.g. 1024)
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


def construct_tasks():
    return {
        "scrolls_qasper": Qasper,
        "scrolls_quality": QuALITY,
        "scrolls_narrativeqa": NarrativeQA,
        "scrolls_contractnli": ContractNLI,
        "scrolls_govreport": GovReport,
        "scrolls_summscreenfd": SummScreenFD,
        "scrolls_qmsum": QMSum,
    }
