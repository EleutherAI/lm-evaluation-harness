import abc
import logging
import re
import datasets
import numpy as np
import promptsource.templates
from abc import abstractmethod
from typing import Callable, List, Mapping, Optional, Tuple, Union

from lm_eval.api import utils
from lm_eval.api.metric import (
    bits_per_byte,
    bleu,
    mean,
    rouge,
    sari,
    weighted_perplexity,
)
from lm_eval.api.request import Request, rf


logger = logging.getLogger(__name__)


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    VERSION = 0

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
    ):
        """
        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.download(data_dir, cache_dir, download_mode)
        self._training_docs = None
        self._fewshot_docs = None

    def download(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
    ):
        """Downloads and returns the task dataset.
        Override this method to download the dataset from a custom API.

        :param data_dir: str
            Stores the path to a local folder containing the `Task`'s data files.
            Use this to specify the path to manually downloaded data (usually when
            the dataset is not publicly accessible).
        :param cache_dir: str
            The directory to read/write the `Task` dataset. This follows the
            HuggingFace `datasets` API with the default cache directory located at:
                `~/.cache/huggingface/datasets`
            NOTE: You can change the cache location globally for a given process
            by setting the shell environment variable, `HF_DATASETS_CACHE`,
            to another directory:
                `export HF_DATASETS_CACHE="/path/to/another/directory"`
        :param download_mode: datasets.DownloadMode
            How to treat pre-existing `Task` downloads and data.
            - `datasets.DownloadMode.REUSE_DATASET_IF_EXISTS`
                Reuse download and reuse dataset.
            - `datasets.DownloadMode.REUSE_CACHE_IF_EXISTS`
                Reuse download with fresh dataset.
            - `datasets.DownloadMode.FORCE_REDOWNLOAD`
                Fresh download and fresh dataset.
        """
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    @abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self) -> datasets.Dataset:
        """
        :return: datasets.Dataset
            A dataset of training documents.
        """
        return datasets.Dataset.from_dict({})

    def validation_docs(self) -> datasets.Dataset:
        """
        :return: datasets.Dataset
            A dataset of validation documents.
        """
        return datasets.Dataset.from_dict({})

    def test_docs(self) -> datasets.Dataset:
        """
        :return: datasets.Dataset
            A dataset of test documents.
        """
        return datasets.Dataset.from_dict({})

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @abstractmethod
    def doc_to_text(self, doc: dict) -> str:
        pass

    @abstractmethod
    def doc_to_target(self, doc: dict) -> str:
        pass

    @abstractmethod
    def construct_requests(self, doc: dict, ctx: str, args: dict) -> List[Request]:
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        :param args: dict
            The specifics of the context, including number of few shots.
        :returns: iterable of `Request` objects.
        """
        pass

    @abstractmethod
    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of sub-metrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abstractmethod
    def aggregation(self) -> Mapping[str, Callable]:
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of sub-metrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abstractmethod
    def higher_is_better(self) -> Mapping[str, bool]:
        """
        :returns: {str: bool}
            A dictionary where keys are the names of sub-metrics and values are
            whether a higher value of the sub-metric is better
        """
        pass


class PromptSourceTask(Task):
    """These are the metrics from promptsource that we have
    added default behavior for. If you want to add default behavior for a new metric,
    update the functions below. If you want to use one of the following metrics,
    *and* add additional custom processing, override `process_results`, `higher_is_better`, and `aggregation`.
    """

    CONFIGURED_RANKED_CHOICE_PS_METRICS = {"Accuracy"}
    CONFIGURED_GENERATION_PS_METRICS = {"BLEU", "ROUGE", "SARI"}
    SPLIT = None

    def __init__(
        self,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        download_mode: Optional[str] = None,
        prompt_template: Optional[promptsource.templates.Template] = None,
        example_separator: Optional[str] = "\n###\n",
        text_target_separator: Optional[str] = " ",
        save_examples: Optional[bool] = True,
    ):
        """
        :param save_examples: Optional[bool]
            Whether to save each example and corresponding model predictions to
            an output `dict`.

        Few-shot Prompting Args:
        :param example_separator: Optional[str]
            The string that will be used to separate the few-shot examples from
            the prompt example.
            Default: "\n###\n"
                See Webson & Pavlick (2022) https://arxiv.org/pdf/2109.01247.pdf
                for justification of this separator.
        :param text_target_separator: Optional[str]
            The string that will be used to separate the prompt example from the
            target text. Example:
            `Q: Where is the Eiffel Tower located? A:{text_target_separator}Paris`
        """
        super().__init__(data_dir, cache_dir, download_mode)
        self.prompt_template = prompt_template
        self.save_examples = save_examples
        self.example_separator = example_separator
        self.text_target_separator = text_target_separator

    def stop_sequences(self) -> List[str]:
        """Denote where the generation should end based on the few-shot example
        separator.

        NOTE: Override this if you want to use a sequence other than just the
        task's few-shot example separator.
        """
        return [self.example_separator]

    def max_generation_length(self) -> Optional[int]:
        """Denote where the max length of the generation if it is obvious from the task."""
        return None

    def evaluation_docs(self) -> datasets.Dataset:
        """Returns the `dataset` split to be used for evaluation."""
        if self.has_test_docs():
            return self.test_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

    def fewshot_docs(self) -> datasets.Dataset:
        """
        Returns the `dataset` split that the few-shot examples should be sample
        from. This prioritizes the `train_docs` split as the few-shot example
        source, then `validation_docs`, and lastly `test_docs`.
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            return self.test_docs()

    def doc_to_text(self, doc: dict) -> str:
        text, _ = self.prompt_template.apply(doc)
        return text

    def doc_to_target(self, doc: dict) -> List[str]:
        _, target = self.prompt_template.apply(doc)
        return target

    def doc_to_rawtext(self, doc: dict) -> str:
        """This should be used for selecting the raw text of the document.

        The current use case is for computing SARI which requires the text
        without the prompt. The `text` field is not standardized across tasks
        so this is task specific.
        """
        raise NotImplementedError("This is task specific.")

    def invalid_doc_for_prompt(self, doc) -> bool:
        """Some prompts may not work for some documents.
        Default: False
        """
        return False

    def format_example(self, text: str, target: str, separator: str) -> str:
        """Returns the text and target combined by the specified `separator`"""
        return text + separator + target

    def fewshot_examples(
        self,
        docs: datasets.Dataset,
        k: int,
        rng: np.random.Generator,
        prompt: dict = None,
    ) -> Tuple[List[dict], List[int]]:
        """Returns `k` random examples from the set of documents in `docs`.

        :param docs: datasets.Dataset
            The dataset of documents to sample few-shot examples from.
        :param k: int
            The number of few-shot examples.
        :param rng: np.random.Generator
            The pseudo-random number generator used to randomly sample examples.
        :param prompt: Optional[dict]
            The prompt document. Specify this to ensure the prompt is not in
            the set of few-shot examples.
        """
        random_indices = np.arange(len(docs)).tolist()
        rng.shuffle(random_indices)

        i = 0
        fewshot_examples, fewshot_idx = [], []
        for idx in random_indices:
            if i >= k:  # Break when we have enough examples.
                break
            if self.invalid_doc_for_prompt(docs[idx]) or docs[idx] == prompt:
                continue
            fewshot_examples.append(docs[idx])
            fewshot_idx.append(int(idx))
            i += 1
        return fewshot_examples, fewshot_idx

    def fewshot_context(
        self, doc: dict, num_fewshot: int, rng: Optional[np.random.Generator]
    ) -> Tuple[str, dict]:
        """Returns a few-shot context string made up of `num_fewshot` number of
        labeled examples, and an appended prompt example without labeling.

        :param doc: dict
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rng: numpy.random.Generator
            The pseudo-random number generator used to randomly sample few-shot examples.
        :returns: Tuple[str, dict]
            ctx: str
                The fewshot context.
            logging_info: dict
                A `dict` of logging info that can be used to identify few-shot sources.
        """
        assert (
            rng is not None
        ), "A `numpy.random.Generator` argument must be provided to `rng`"

        if num_fewshot == 0:
            labeled_examples = ""
            fewshot_idx, fewshot_target_idx, fewshot_src = ([], [], None)
        else:
            # Construct few-shot labeled examples.
            fewshot_docs = self.fewshot_docs()
            fewshot_src = str(fewshot_docs.split)
            fewshot_examples, fewshot_idx = self.fewshot_examples(
                fewshot_docs, k=num_fewshot, rng=rng, prompt=doc
            )
            labeled_examples_list = []
            fewshot_target_idx = []
            for fewshot_example in fewshot_examples:
                text = self.doc_to_text(fewshot_example)
                targets = self.doc_to_target(fewshot_example)
                # Choose 1 random target from multi-reference targets.
                target_idx = int(rng.integers(0, len(targets)))
                target = targets[target_idx].strip()
                labeled_examples_list.append(
                    self.format_example(text, target, self.text_target_separator)
                )
                fewshot_target_idx.append(target_idx)
            labeled_examples = self.example_separator.join(labeled_examples_list)
            # Leave an extra `example_separator` right before the prompt.
            labeled_examples += self.example_separator

        prompt = self.doc_to_text(doc)
        ctx = labeled_examples + prompt
        logging_info = {
            "fewshot_idx": fewshot_idx,
            "fewshot_target_idx": fewshot_target_idx,
            "fewshot_source": fewshot_src,
            "fewshot_num": num_fewshot,
            "ctx": ctx,
        }
        return ctx, logging_info

    def construct_requests(self, doc: dict, ctx: str, args: dict) -> List[Request]:
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        :param args: dict
            The specifics of the context, including number of few shots.
        :returns: iterable of `Request` objects.
        """
        requests = []
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        if answer_choices_list:
            # If answer_choices_list, then this is a ranked choice prompt.
            for answer_choice in answer_choices_list:
                ll_answer_choice, _ = rf.loglikelihood(
                    ctx, self.text_target_separator + answer_choice
                )
                requests.append(ll_answer_choice)
        else:
            # If not, then this is a generation prompt.
            request_args = {
                "stop_sequences": self.stop_sequences(),
                "max_generation_length": self.max_generation_length(),
                "num_fewshot": args["num_fewshot"],
            }
            cont_request = rf.greedy_until(ctx, request_args)
            requests.append(cont_request)
        return requests

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of sub-metrics and values are the values of
        the metric for that one document

        NOTE: This function automates processing by using the `promptsource`
        metadata to determine the metric.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        target = self.doc_to_target(doc)
        if answer_choices_list:
            # If answer_choices_list, then this is a ranked choice prompt.
            # NOTE: In the future, target could be a list of strings.
            assert isinstance(target, list) and len(target) == 1
            target = target[0].strip()
            target_idx = answer_choices_list.index(target)

            pred = answer_choices_list[np.argmax(results)]
            out = {}

            for metric in self.prompt_template.metadata.metrics:
                if metric not in self.CONFIGURED_RANKED_CHOICE_PS_METRICS:
                    logger.warning(
                        f"Unexpected metric: `{metric}`. Add it, or use a task-specific solution."
                    )
                if metric == "Accuracy":
                    out["acc"] = pred == target
                    # Byte-length normalization.
                    completion_len = np.array(
                        [float(len(i)) for i in answer_choices_list]
                    )
                    out["acc_norm"] = (
                        1.0
                        if np.argmax(results / completion_len) == target_idx
                        else 0.0
                    )
            # TODO: Add metrics here.
        else:
            # If not, then this is a generation prompt.
            # NOTE: In the future, target will be a list of strings.
            assert isinstance(target, list)
            pred = results[0].strip()
            out = {}
            for metric in self.prompt_template.metadata.metrics:
                if metric not in self.CONFIGURED_GENERATION_PS_METRICS:
                    logger.warning(
                        f"Unexpected metric: `{metric}`. Add it, or use a task-specific solution."
                    )
                if metric == "BLEU":
                    out["bleu"] = (target, pred)
                elif metric == "ROUGE":
                    # TODO: This computes all rouge sub-metrics. Find a generic
                    # way to handle user specified rouge sub-metrics to avoid extra
                    # compute.
                    rouge_scores = rouge(target, pred)
                    # Flatten rouge score dict.
                    rouge_scores = utils.flatten(rouge_scores)
                    # Merge all the rouge-type scores into the `out` dict.
                    out = {**out, **rouge_scores}
                elif metric == "SARI":
                    out["sari"] = sari(self.doc_to_rawtext(doc), pred, target)

        # TODO: Wrap process results s.t. override impl do not
        # override the save examples.
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
                "answer_choices_list": answer_choices_list,
            }
            return out, example
        return out

    def aggregation(self) -> Mapping[str, Callable]:
        out = {}
        for metric in self.prompt_template.metadata.metrics:
            if metric == "Accuracy":
                out["acc"] = mean
                out["acc_norm"] = mean
            elif metric == "BLEU":
                out["bleu"] = bleu
            elif metric == "ROUGE":
                # TODO: Find a generic way to handle user specified rouge metrics.
                out["rouge1_precision"] = mean
                out["rouge1_recall"] = mean
                out["rouge1_fmeasure"] = mean

                out["rouge2_precision"] = mean
                out["rouge2_recall"] = mean
                out["rouge2_fmeasure"] = mean

                out["rougeL_precision"] = mean
                out["rougeL_recall"] = mean
                out["rougeL_fmeasure"] = mean

                out["rougeLsum_precision"] = mean
                out["rougeLsum_recall"] = mean
                out["rougeLsum_fmeasure"] = mean
            elif metric == "SARI":
                out["sari"] = mean
        return out

    def higher_is_better(self) -> Mapping[str, bool]:
        out = {}
        for metric in self.prompt_template.metadata.metrics:
            if metric == "Accuracy":
                out["acc"] = True
                out["acc_norm"] = True
            elif metric == "BLEU":
                out["bleu"] = True
            elif metric == "ROUGE":
                # TODO: Find a generic way to handle user specified rouge metrics.
                out["rouge1_precision"] = True
                out["rouge1_recall"] = True
                out["rouge1_fmeasure"] = True

                out["rouge2_precision"] = True
                out["rouge2_recall"] = True
                out["rouge2_fmeasure"] = True

                out["rougeL_precision"] = True
                out["rougeL_recall"] = True
                out["rougeL_fmeasure"] = True

                out["rougeLsum_precision"] = True
                out["rougeLsum_recall"] = True
                out["rougeLsum_fmeasure"] = True
            elif metric == "SARI":
                out["sari"] = True
        return out

    def get_logging_info(self):
        return {
            "fixed_answer_choice_list": self.prompt_template.get_fixed_answer_choices_list(),
            "dataset_path": self.DATASET_PATH,
            "dataset_name": self.DATASET_NAME,
            "subset": self.SPLIT,
            "prompt_name": self.prompt_template.get_name(),
            "prompt_id": self.prompt_template.get_id(),
            "prompt_jinja": self.prompt_template.jinja,
            "prompt_original_task": self.prompt_template.metadata.original_task,
            # Placeholder for comment in post-processing.
            "comment": "",
        }


class TranslationTask(PromptSourceTask):

    # Language specific functions.
    @classmethod
    def zh_split(cls, zh_text: str) -> List[str]:
        """Chinese splitting"""
        import jieba

        return [" ".join(jieba.cut(txt.strip())) for txt in zh_text]

    @classmethod
    def ja_split(cls, ja_text: str) -> List[str]:
        """Japanese splitting"""
        import nagisa

        return [" ".join(nagisa.tagging(txt.strip()).words) for txt in ja_text]

    NO_SPACE_LANG = {"zh": zh_split, "ja": ja_split}

    def invalid_doc_for_prompt(self, doc) -> bool:
        # Skip docs with empty references.
        if self.doc_to_target(doc) == [""]:
            return True
        return False

    def _get_src_ref_codes(self, template_name: str) -> Tuple[str, str]:
        """Returns a 2-tuple of (src_lang, ref_lang) codes from the prompt template name."""
        # Get the lang codes from the dataset name.
        lang_pairs = self.DATASET_NAME.split("-")
        # Template name ordering defines the src and ref lang codes.
        if self.DATASET_NAME in template_name:
            return lang_pairs[0], lang_pairs[1]
        # Flip the lang pairs following the prompt source.
        return lang_pairs[1], lang_pairs[0]

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        answer_choices_list = self.prompt_template.get_answer_choices_list(doc)
        target = self.doc_to_target(doc)

        # Add spaces between words for BLEU score calculation of target languages like Chinese
        _, tar_lang_code = self._get_src_ref_codes(self.prompt_template.name)
        if tar_lang_code in self.NO_SPACE_LANG:
            target = [self.NO_SPACE_LANG[tar_lang_code]([t])[0] for t in target]
            results = self.NO_SPACE_LANG[tar_lang_code](results)
        pred = results[0].strip()

        out = {}
        for metric in self.prompt_template.metadata.metrics:
            assert (
                metric in self.CONFIGURED_GENERATION_PS_METRICS
            ), "Unexpected metric. Add it, or use a task-specific solution."
            if metric == "BLEU":
                out["bleu"] = (target, pred)
            elif metric == "ROUGE":
                # TODO: This computes all rouge sub-metrics. Find a generic
                # way to handle user specified rouge sub-metrics to avoid extra
                # compute.
                rouge_scores = rouge(target, pred)
                # Flatten rouge score dict.
                rouge_scores = utils.flatten(rouge_scores)
                # Merge all the rouge-type scores into the `out` dict.
                out = {**out, **rouge_scores}

        # TODO: Wrap process results s.t. override impl do not
        # override the save examples.
        if self.save_examples:
            example = {
                "pred": pred,
                "target": target,
                "answer_choices_list": answer_choices_list,
            }
            return out, example
        return out


class PerplexityTask(PromptSourceTask):
    """NOTE: Prompts are ignored for perplexity tasks."""

    def doc_to_text(self, doc: dict) -> str:
        return ""

    def doc_to_target(self, doc: dict) -> List[str]:
        """Because prompts are ignored, return the relevant text from doc."""
        raise NotImplementedError()

    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[str, dict]:
        assert (
            num_fewshot == 0
        ), "The number of fewshot examples must be 0 for perplexity tasks."
        assert (
            rng is not None
        ), "A `numpy.random.Generator` argument must be provided to `rng`"
        return (
            "",
            {
                "fewshot_idx": [],
                "fewshot_target_idx": [],
                "fewshot_source": None,
                "fewshot_num": 0,
                "ctx": "",
            },
        )

    def construct_requests(self, doc: dict, ctx: str, args: dict) -> List[Request]:
        assert not ctx
        string = self.doc_to_target(doc)[0]
        req = rf.loglikelihood_rolling(string)
        return req

    def process_results(
        self, doc: dict, results: list
    ) -> Union[dict, Tuple[dict, dict]]:
        (loglikelihood,) = results
        target = self.doc_to_target(doc)[0]
        words = self.count_words(target)
        bytes_ = self.count_bytes(target)

        out = {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }
        if self.save_examples:
            return out, {
                "word_perplexity_instance": weighted_perplexity(
                    [(loglikelihood, words)]
                ),
                "byte_perplexity_instance": weighted_perplexity(
                    [(loglikelihood, bytes_)]
                ),
                "bits_per_byte_instance": bits_per_byte([(loglikelihood, bytes_)]),
            }
        return out

    def aggregation(self) -> Mapping[str, Callable]:
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    def higher_is_better(self) -> Mapping[str, bool]:
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    @classmethod
    def count_bytes(cls, doc):
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """Downstream tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))

    def get_logging_info(self):
        return {
            "prompt_name": None,
        }
