from __future__ import annotations

import abc
import ast
import logging
import random
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property, partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import datasets
import numpy as np
from tqdm import tqdm
from typing_extensions import deprecated

from lm_eval import utils
from lm_eval.api.instance import Instance, OutputType
from lm_eval.api.samplers import ContextSampler
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.metric import MetricConfig
from lm_eval.config.task import DataSet, TaskConfig
from lm_eval.filters import build_filter_ensemble
from lm_eval.utils import validate_index


ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "generate_until",
]

if TYPE_CHECKING:
    pass


eval_logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


def format_turn(content: str, role: str):
    return {"role": role, "content": content}


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    VERSION: int | str | None = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str | None = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str | None = None

    OUTPUT_TYPE: OutputType | None = None

    def __init__(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode: datasets.DownloadMode | None = None,
        config: Mapping | None = None,  # Union[dict, TaskConfig]
    ) -> None:
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
        self._training_docs: list | None = None
        self._fewshot_docs: list | None = None
        self._instances: list[Instance] | None = None

        self._config: TaskConfig = TaskConfig.from_yaml({**config})

        self._filters = [build_filter_ensemble("none", [("take_first", None)])]
        self.fewshot_rnd: random.Random | None = (
            None  # purposely induce errors in case of improper usage
        )
        self.sampler = ContextSampler(list(self.fewshot_docs))
        self.multiple_input = False

    def download(
        self,
        data_dir: str | None = None,
        cache_dir: str | None = None,
        download_mode=None,
    ) -> None:
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
        assert self.DATASET_PATH is not None, "DATASET_PATH must be set in Task class"
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    @property
    def config(self) -> TaskConfig:
        """Returns the TaskConfig associated with this class."""
        return self._config

    @property
    def has_training_docs(self) -> bool:
        """Whether the task has a training set"""
        raise NotImplementedError

    @property
    def has_validation_docs(self) -> bool:
        """Whether the task has a validation set"""
        raise NotImplementedError

    @property
    def has_test_docs(self) -> bool:
        """Whether the task has a test set"""
        raise NotImplementedError

    def training_docs(self) -> DataSet | None:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self) -> DataSet | None:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self) -> DataSet | None:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self) -> DataSet | None:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs:
            return self.training_docs()
        elif self.has_validation_docs:
            return self.validation_docs()
        else:
            if self.config.num_fewshot and self.config.num_fewshot > 0:
                eval_logger.warning(
                    f"[Task: {self.config.task}] has_training_docs and has_validation_docs are False"
                    ", using test_docs as fewshot_docs but this is not recommended."
                )
            return self.test_docs()

    def _process_doc(self, doc: dict) -> dict:
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def instances(self) -> list[Instance]:
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    def fewshot_examples(self, k: int, rnd) -> Iterable[dict]:
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc: dict):
        raise NotImplementedError(
            "Override doc_to_decontamination_query with document specific decontamination query."
        )

    @abc.abstractmethod
    def doc_to_text(self, doc: dict) -> str:
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc: dict) -> str | int:
        pass

    # not an abstractmethod because not every language-only task has to implement this
    def doc_to_image(self, doc: dict):
        raise NotImplementedError

    def doc_to_audio(self, doc: dict):
        raise NotImplementedError

    @staticmethod
    def resolve_field(doc: dict[str, str], field: str | None = None):
        if field is not None:
            return doc[field] if field in doc else utils.apply_template(field, doc)

    def build_all_requests(
        self,
        *,
        limit: int | None = None,
        samples: list[int] | None = None,
        rank: int = 0,
        world_size: int = 1,
        cache_requests: bool = False,
        rewrite_requests_cache: bool = False,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable | None = None,
        tokenizer_name: str = "",
    ) -> None:
        """Build a set of Instances for a task, and store them in task.instances"""

        # used with caching
        og_limit = limit

        cache_key = f"requests-{self._config.task}-{self.config.num_fewshot}shot-rank{rank}-world_size{world_size}"
        cache_key += "-chat_template" if apply_chat_template else ""
        cache_key += "-fewshot_as_multiturn" if fewshot_as_multiturn else ""
        cache_key += (
            f"-system_prompt_hash{utils.hash_string(system_instruction)}"
            if system_instruction is not None
            else ""
        )
        cache_key += f"-tokenizer{tokenizer_name}"

        cached_instances = load_from_cache(file_name=cache_key, cache=cache_requests)

        if cache_requests and cached_instances and not rewrite_requests_cache:
            cached_instances = cached_instances[:limit]

            flattened_instances = [
                instance
                for instance_group in cached_instances
                for instance in instance_group
            ]

            self._instances = flattened_instances
            return

        eval_logger.info(f"Building contexts for {self.config.task} on rank {rank}...")

        instances = []

        # process all documents when caching is specified for simplicity
        if (
            cache_requests
            and (not cached_instances or rewrite_requests_cache)
            and limit is not None
        ):
            limit = None

        doc_id_docs = list(
            self.doc_iterator(
                rank=rank, limit=limit, samples=samples, world_size=world_size
            )
        )

        num_docs = len(doc_id_docs)

        for doc_id, doc in tqdm(
            doc_id_docs,
            total=num_docs,
        ):
            # sample fewshot context #TODO: need to offset doc_id by rank now!
            fewshot_ctx = self.fewshot_context(
                doc,
                num_fewshot=0
                if self.config.num_fewshot is None
                else self.config.num_fewshot,
                system_instruction=system_instruction,
                apply_chat_template=apply_chat_template,
                fewshot_as_multiturn=fewshot_as_multiturn,
                chat_template=chat_template,
                gen_prefix=self.resolve_field(doc, self.config.gen_prefix),
            )

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
            inst = self.construct_requests(
                doc=doc,
                ctx=fewshot_ctx,
                metadata=(self.config.task, doc_id, self.config.repeats),
                apply_chat_template=apply_chat_template,
                chat_template=chat_template,
            )

            if not isinstance(inst, list):
                inst = [inst]

            instances.append(inst)

        # now flatten, this is to allow slicing to work with pickles

        sliced_instances = instances[:og_limit]

        flattened_instances = [
            instance
            for instance_group in sliced_instances
            for instance in instance_group
        ]

        self._instances = flattened_instances

        if len(self._instances) == 0:
            raise ValueError("task.build_requests() did not find any docs!")

        if cache_requests and (not cached_instances or rewrite_requests_cache):
            save_to_cache(file_name=cache_key, obj=instances)

    @abc.abstractmethod
    def construct_requests(self, doc: dict, ctx: list[dict] | str, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        :param doc_idx: int
            The index of a document within `self.test_docs()` or `self.validation_docs()`,
            whichever is the main split used.
        :param repeats: int
        TODO: update this docstring
            The number of times each instance in a dataset is inferred on. Defaults to 1,
            can be increased for techniques like majority voting.
        """

    @abc.abstractmethod
    def process_results(self, doc: dict, results: list) -> dict[str, Any]:
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        raise NotImplementedError

    @deprecated("not used anymore")
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return True

    @deprecated("not used anymore")
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return True

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @staticmethod
    def count_bytes(doc: str) -> int:
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @staticmethod
    def count_words(doc: str) -> int:
        """Downstream loglikelihood_rolling perplexity tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))

    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, rnd=None, description=None, **kwargs):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        if rnd is None:
            if self.fewshot_rnd is not None:
                rnd = self.fewshot_rnd
            else:
                raise ValueError(
                    "A `random.Random` generator argument must be provided to `rnd`"
                )

        description = description if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs:
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs
                        else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = (
                "\n\n".join(
                    [
                        self.doc_to_text(doc) + self.doc_to_target(doc)
                        for doc in fewshotex
                    ]
                )
                + "\n\n"
            )

        example = self.doc_to_text(doc)
        return description + labeled_examples + example

    def apply_filters(self) -> list[Instance] | None:
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters") and self._instances:
            for f in self._filters:
                f.apply(self._instances)
        else:
            eval_logger.warning(
                "No filter defined or no instances, passing through instances"
            )
            return self._instances

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (num_fewshot)
        return self.config.to_dict()

    def set_config(self, key: str, value: Any, update: bool = False) -> None:
        """Set or update the configuration for a given key."""
        if update:
            current_value = getattr(self._config, key, {})
            if not isinstance(current_value, dict):
                raise TypeError(
                    f"Expected a dict for key '{key}', got {type(current_value).__name__} instead."
                )
            current_value.update(value)
        else:
            setattr(self._config, key, value)

    def override_metric(self, metric_name: str) -> None:
        """
        Override the default metrics used for evaluation with custom metrics.

        Parameters:
        - metric_name (str): The name of the custom metric to override. Should be registered in api.metrics.
        """
        # if not isinstance(self, ConfigurableTask):
        #     self.process_results = lambda x, y: {metric_name: get_metric(metric_name)}
        #     self.aggregation = lambda: {
        #         metric_name: get_metric_aggregation(metric_name)
        #     }
        self._config.metric_list = [MetricConfig(name=metric_name)]
        self._config.process_results = lambda *args: {"bypass": 0}

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        if hasattr(self, "sampler"):
            self.sampler.set_rnd(seed)

    @property
    def eval_docs(self) -> datasets.Dataset | Iterable[dict]:
        if self.has_test_docs:
            return self.test_docs()
        elif self.has_validation_docs:
            return self.validation_docs()
        else:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"
            )

    def doc_iterator(
        self,
        *,
        rank: int = 0,
        limit: int | None = None,
        world_size: int = 1,
        samples: list[int] | None = None,
    ) -> Iterator[tuple[int, Any]]:
        if samples:
            n = len(self.eval_docs)
            assert all(e < n for e in samples), (
                f"Elements of --samples should be in the interval [0,k-1] where k is the number of total examples. In this case, k={n}."
            )
            eval_logger.info(
                f"{self.config.task}: Evaluating on {len(samples)} examples"
            )
            doc_iterator = utils.create_iterator(
                enumerate(x for i, x in enumerate(self.eval_docs) if i in samples),
                rank=int(rank),
                limit=None,  # limit does not matter here since we are selecting samples directly
                world_size=int(world_size),
            )
        else:
            limit = int(limit) if limit else None
            doc_iterator = utils.create_iterator(
                enumerate(self.eval_docs),
                rank=int(rank),
                limit=limit,
                world_size=int(world_size),
            )
        return doc_iterator


class ConfigurableTask(Task):
    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        # Get pre-configured attributes
        self._config = self.CONFIG
        self.fewshot_rnd = 1234

        # Use new configurations if there was no preconfiguration
        if self.config is None:
            self._config = TaskConfig.from_yaml(config)
        # Overwrite configs
        else:
            if config is not None:
                self._config.__dict__.update(config)

        if self.config is None:
            raise ValueError(
                "Must pass a config to ConfigurableTask, either in cls.CONFIG or `config` kwarg"
            )

        if isinstance(self.config.metadata, dict) and "version" in self.config.metadata:
            self.VERSION = self.config.metadata["version"]

        if self.config.output_type is not None:
            if self.config.output_type not in ALL_OUTPUT_TYPES:
                raise ValueError(
                    f"Got invalid output_type '{self.config.output_type}', must be in '{','.join(ALL_OUTPUT_TYPES)}'"
                )
            self.OUTPUT_TYPE = self.config.output_type

        self.multiple_targets = self.config.multiple_targets
        self.multiple_inputs = self.config.multiple_inputs
        assert not (self.multiple_targets and self.multiple_inputs), (
            "Cannot have both multiple_targets and multiple_inputs"
        )

        if self.config.doc_to_image is not None:
            # mark the task as requiring multimodality.
            self.MULTIMODAL = True

        if self.config.doc_to_audio:
            # mark the task as requiring multimodality.
            self.MULTIMODAL = True

        if self.config.unsafe_code is not False:
            self.UNSAFE_CODE = True

        if self.config.dataset_path is not None:
            self.DATASET_PATH = self.config.dataset_path

        if self.config.dataset_name is not None:
            self.DATASET_NAME = self.config.dataset_name

        # self.metric_list: list[MetricConfig] = self.config.get_metrics

        self.download(self.config.dataset_kwargs)
        self._training_docs = None
        self._fewshot_docs = None

        self._filters = self.config.get_filters

        # if self.config.use_prompt is not None:
        #     eval_logger.info(f"loading prompt {self.config.use_prompt}")
        #     self.prompt = get_prompt(
        #         self.config.use_prompt, self.DATASET_PATH, self.DATASET_NAME
        #     )
        # else:
        #     self.prompt = None

        if (
            self.config.fewshot_cfg.num_fewshot() > 0
            and self.fewshot_docs() is not None
        ):
            self.sampler = self.config.fewshot_cfg.init_sampler(
                list(self.fewshot_docs()), self, rnd=self.fewshot_rnd
            )
        self.task_docs = self.eval_docs

        # for name, fn in self.config._fn.items():
        #     if hasattr(self, name):
        #         setattr(
        #             self,
        #             name,
        #             types.MethodType(
        #                 lambda self, *args, _fn=fn, **kwargs: _fn(*args, **kwargs),
        #                 self,
        #             ),
        #         )

        self.runtime_checks(self.task_docs[0])

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        from packaging.version import parse as vparse

        self.config.dataset_kwargs, self.config.metadata = (
            self.config.dataset_kwargs or {},
            self.config.metadata or {},
        )
        if dataset_kwargs and vparse(datasets.__version__) >= vparse("4.0.0"):
            dataset_kwargs.pop("trust_remote_code", None)
        if isinstance(df := self.config.custom_dataset, Callable):
            eval_logger.warning(
                f"{self.config.task}: Custom kwargs can be passed to `--metadata` in console (as json string) or to the TaskManager."
                + "\nFor example --metadata='{\"max_seq_lengths\":[4096, 8192]}'. For details see task Readme."
            )
            self.dataset = df(**(self.config.dataset_kwargs | self.config.metadata))
        else:
            assert self.config.dataset_path is not None, (
                "dataset_path must be set in TaskConfig"
            )
            self.dataset = datasets.load_dataset(
                path=self.config.dataset_path,
                name=self.config.dataset_name,
                **self.config.dataset_kwargs,
            )

    @cached_property
    def has_training_docs(self) -> bool:
        return self.config.training_split is not None

    @cached_property
    def has_validation_docs(self) -> bool:
        return self.config.validation_split is not None

    @cached_property
    def has_test_docs(self) -> bool:
        return self.config.test_split is not None

    def training_docs(self) -> DataSet | None:
        if self.has_training_docs:
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.training_split]
                )
            return self.dataset[self.config.training_split]

    def validation_docs(self) -> DataSet | None:
        if self.has_validation_docs:
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.validation_split]
                )
            return self.dataset[self.config.validation_split]

    def test_docs(self) -> DataSet | None:
        if self.has_test_docs:
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.test_split])
            return self.dataset[self.config.test_split]

    def fewshot_docs(self):
        docs = self.config.fewshot_cfg.get_docs(self.dataset)

        if docs is not None:
            return docs

        # Fallback to parent implementation
        if (
            (_num_fewshot := self.config.num_fewshot)
            and isinstance(_num_fewshot, int)
            and _num_fewshot > 0
        ):
            eval_logger.warning(
                f"[Task: {self.config.task}] "
                "num_fewshot > 0 but no fewshot source configured. "
                "Using preconfigured rule."
            )

        return super().fewshot_docs()

    def apply_filters(self) -> list[Instance] | None:
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters") and self._instances:
            for f in self._filters:
                f.ensemble.apply(self._instances)
        else:
            eval_logger.warning(
                "No filter defined or instances found. Passing through instances"
            )
        return self._instances

    def should_decontaminate(self):
        return self.config.should_decontaminate

    def doc_to_decontamination_query(self, doc: dict):
        if self.config.should_decontaminate:
            if self.config.doc_to_decontamination_query is None:
                return self.doc_to_text(doc)
            else:
                doc_to_decontamination_query = self.config.doc_to_decontamination_query
                if doc_to_decontamination_query in self.features:
                    return doc[doc_to_decontamination_query]
                elif callable(doc_to_decontamination_query):
                    return doc_to_decontamination_query(doc)
                else:
                    return ast.literal_eval(
                        utils.apply_template(
                            self.config.doc_to_decontamination_query, doc
                        )
                    )

    def _process_doc(self, doc: dict) -> dict:
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @overload
    def doc_to_text(self, doc: dict, doc_to_text: None = None) -> str | int: ...

    @overload
    def doc_to_text(self, doc: dict, doc_to_text: int) -> int: ...

    @overload
    def doc_to_text(self, doc: dict, doc_to_text: str) -> str: ...

    @overload
    def doc_to_text(self, doc: dict, doc_to_text: Callable[..., str]) -> str: ...

    def doc_to_text(
        self, doc: dict, doc_to_text: int | str | Callable[..., str] | None = None
    ) -> str | int:
        # if self.prompt is not None:
        #     doc_to_text = self.prompt
        doc_to_text = doc_to_text or self.config.doc_to_text
        if callable(doc_to_text):
            return doc_to_text(doc)
        if doc_to_text in doc:
            return doc[doc_to_text]
        elif isinstance(doc_to_text, str):
            text_string = utils.apply_template(doc_to_text, doc)
            if text_string.isdigit() and self.config.doc_to_choice is not None:
                return ast.literal_eval(text_string)
            else:
                return text_string
        elif isinstance(doc_to_text, int):
            return doc_to_text
        # Used when applying a Promptsource template
        # elif hasattr(doc_to_text, "apply"):
        #     applied_prompt = doc_to_text.apply(doc)
        #     if len(applied_prompt) == 2:
        #         return applied_prompt[0]
        #     else:
        #         eval_logger.warning("Applied prompt returns empty string")
        #         return self.config.fewshot_delimiter
        else:
            print(type(doc_to_text))
            raise TypeError

    @overload
    def doc_to_target(
        self, doc: dict, doc_to_target: None = None
    ) -> int | str | list[int]: ...

    @overload
    def doc_to_target(self, doc: dict, doc_to_target: int) -> int: ...

    @overload
    def doc_to_target(self, doc: dict, doc_to_target: str) -> int | str | list[int]: ...

    @overload
    def doc_to_target(self, doc: dict, doc_to_target: list) -> list[int]: ...

    @overload
    def doc_to_target(
        self, doc: dict, doc_to_target: Callable[..., int | str | list[int]]
    ) -> int | str | list[int]: ...

    def doc_to_target(self, doc: dict, doc_to_target=None) -> int | str | list[int]:
        # if self.prompt is not None:
        #     doc_to_target = self.prompt
        doc_to_target = doc_to_target or self.config.doc_to_target
        if callable(doc_to_target):
            doc_to_target(doc)
        if doc_to_target in doc:
            return doc[doc_to_target]
        elif isinstance(doc_to_target, str):
            target_string = utils.apply_template(doc_to_target, doc)
            if target_string.isdigit() and self.config.doc_to_choice is not None:
                return ast.literal_eval(target_string)
            # elif (
            #     len(target_string) >= 2
            #     and (target_string[0] == "[")
            #     and (target_string[-1] == "]")
            # ):
            #     try:
            #         return ast.literal_eval(target_string)
            #     except (SyntaxError, ValueError):
            #         return target_string
            else:
                return target_string

        elif isinstance(doc_to_target, (int, list)):
            return doc_to_target
        # elif isinstance(doc_to_target, list):
        #     return doc_to_target
        # elif callable(doc_to_target):
        #     return doc_to_target(doc)
        # # Used when applying a Promptsource template
        # elif hasattr(doc_to_target, "apply"):
        #     applied_prompt = doc_to_target.apply(doc)
        #     if len(applied_prompt) == 2:
        #         return applied_prompt[1]
        #     else:
        #         eval_logger.warning("Applied prompt returns empty string")
        #         return self.config.fewshot_delimiter
        else:
            raise TypeError

    @overload
    def doc_to_choice(self, doc: dict, doc_to_choice: None = None) -> list[str]: ...

    @overload
    def doc_to_choice(self, doc: dict, doc_to_choice: str) -> list[str]: ...

    @overload
    def doc_to_choice(self, doc: dict, doc_to_choice: list) -> list[str]: ...

    @overload
    def doc_to_choice(self, doc: dict, doc_to_choice: dict) -> list[str]: ...

    @overload
    def doc_to_choice(
        self, doc: dict, doc_to_choice: Callable[..., list[str]]
    ) -> list[str]: ...

    def doc_to_choice(
        self,
        doc: dict,
        doc_to_choice: str | list | dict | Callable[..., list[str]] | None = None,
    ) -> list[str]:
        # if self.prompt is not None:
        #     doc_to_choice = self.prompt
        if doc_to_choice is not None:
            doc_to_choice = doc_to_choice
        elif self.config.doc_to_choice is None:
            eval_logger.error("doc_to_choice was called but not set in config")
            doc_to_choice = None
        else:
            doc_to_choice = self.config.doc_to_choice

        if isinstance(doc_to_choice, str):
            if doc_to_choice in doc:
                return doc[doc_to_choice]
            else:
                return ast.literal_eval(utils.apply_template(doc_to_choice, doc))
        elif isinstance(doc_to_choice, list):
            return doc_to_choice
        # elif isinstance(doc_to_choice, dict):
        #     return list(doc_to_choice.values())
        # elif hasattr(doc_to_choice, "get_answer_choices_list"):
        #     return doc_to_choice.get_answer_choices_list(doc)
        else:
            raise TypeError

    @overload
    def doc_to_image(self, doc: dict, doc_to_image: None = None) -> None: ...

    @overload
    def doc_to_image(self, doc: dict, doc_to_image: list) -> list: ...

    @overload
    def doc_to_image(self, doc: dict, doc_to_image: str) -> int | str | None: ...

    @overload
    def doc_to_image(self, doc: dict, doc_to_image: Callable[..., Any]) -> Any: ...

    def doc_to_image(self, doc: dict, doc_to_image=None) -> int | str | list | None:
        if doc_to_image is not None:
            doc_to_image = doc_to_image
        elif self.config.doc_to_image is not None:
            doc_to_image = self.config.doc_to_image
        else:
            return None

        if isinstance(doc_to_image, list):
            image_feature = [
                self.doc_to_image(doc, feature) for feature in doc_to_image
            ]
            return [feature for feature in image_feature if feature is not None]
        elif isinstance(doc_to_image, str):
            if doc_to_image in self.features:
                return doc[doc_to_image]
            else:
                return ast.literal_eval(utils.apply_template(doc_to_image, doc))
        elif callable(doc_to_image):
            return doc_to_image(doc)
        else:
            return None

    @overload
    def doc_to_audio(self, doc: Any, doc_to_audio: None = None) -> None: ...

    @overload
    def doc_to_audio(self, doc: Any, doc_to_audio: list) -> list: ...

    @overload
    def doc_to_audio(self, doc: Any, doc_to_audio: str) -> int | str | None: ...

    @overload
    def doc_to_audio(self, doc: Any, doc_to_audio: Callable[..., Any]) -> Any: ...

    def doc_to_audio(self, doc: Any, doc_to_audio=None) -> int | str | list | None:
        if doc_to_audio is not None:
            doc_to_audio = doc_to_audio
        elif self.config.doc_to_audio is not None:
            doc_to_audio = self.config.doc_to_audio
        else:
            return None

        if isinstance(doc_to_audio, list):
            audio_feature = [
                self.doc_to_audio(doc, feature) for feature in doc_to_audio
            ]
            return [feature for feature in audio_feature if feature is not None]
        elif isinstance(doc_to_audio, str):
            if doc_to_audio in self.features:
                return doc[doc_to_audio]
            else:
                return ast.literal_eval(utils.apply_template(doc_to_audio, doc))
        elif callable(doc_to_audio):
            return doc_to_audio(doc)
        else:
            return None

    def _doc_to_qa_pair(
        self,
        doc: dict[str, Any],
        gen_prefix: str | None,
        *,
        q: str | None = None,
        a: str | None = None,
        include_answer: bool = True,
    ) -> list[Message]:
        """Return `[user, assistant?]` for a single doc."""
        q = q or self.doc_to_text(doc)
        a = a or self.doc_to_target(doc)
        # Handle multiple-choice indirection
        if isinstance(q, list) and self.config.doc_to_choice:
            q = q[cast(int, self.doc_to_target(doc))]
        if isinstance(a, int) and self.config.doc_to_choice:
            a = (
                self.doc_to_choice(doc)[a]
                if not self.multiple_inputs
                else self.doc_to_choice(doc)[0]
            )

        assert isinstance(q, str), "Context is not a string!"
        msgs = [Message("user", q)]
        if include_answer:
            if gen_prefix and not gen_prefix[-1].isspace():
                prefix = gen_prefix + " "
            elif gen_prefix:
                prefix = gen_prefix
            else:
                prefix = ""
            answer_txt = prefix + (a if not isinstance(a, list) else a[0])
            msgs.append(Message("assistant", answer_txt))
        else:
            msgs.append(Message("assistant", gen_prefix)) if gen_prefix else None
        return msgs

    @staticmethod
    def _render_chat_template(
        messages: list[Message],
        chat_template: Callable[[list[dict[str, str]]], str],
        *,
        tgt_delim: str = " ",
        few_delim: str = "\n\n",
        multiturn=True,
    ) -> str:
        if multiturn:
            return chat_template([m.__dict__ for m in messages])
        else:
            has_prefix = messages[-1].role == "assistant"
            if not has_prefix:
                context = [
                    format_turn(
                        ConfigurableTask._message_to_text(
                            messages, tgt_delim=tgt_delim, few_delim=few_delim
                        ),
                        role="user",
                    )
                ]
            else:
                context = [
                    format_turn(
                        ConfigurableTask._message_to_text(
                            messages[:-1], tgt_delim=tgt_delim, few_delim=few_delim
                        ),
                        role="user",
                    )
                ]
                context += [format_turn(**messages[-1].__dict__)]

            return chat_template(context)

    def fewshot_context(
        self,
        doc: dict[str, str],
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable[..., str] | None = None,
        gen_prefix: str | None = None,
    ) -> str | list[str]:
        messages = []
        tgt_delim, few_delim = (
            self.config.target_delimiter,
            self.config.fewshot_delimiter,
        )
        chat_template = (
            partial(chat_template, add_generation_prompt=not gen_prefix)
            if chat_template
            else None
        )
        description = self.resolve_field(doc, self.config.description) or ""
        system_prompt = few_delim.join(filter(None, [system_instruction, description]))
        if system_prompt:
            messages.append(Message("system", system_prompt))

        for fs_doc in self.sampler.sample(
            n=num_fewshot,
            doc=doc if self.config.fewshot_split == self.config.test_split else None,
        ):
            messages += self._doc_to_qa_pair(fs_doc, gen_prefix)

        if self.multiple_inputs:
            # if multiple inputs, then doc_to_text: list[str]
            messages = [
                messages
                + self._doc_to_qa_pair(
                    doc,
                    gen_prefix,
                    q=q,
                    include_answer=False,
                )
                for q in cast(list[str], self.doc_to_text(doc))
            ]
        else:
            # otherwise, doc_to_text: str for all other cases
            messages += self._doc_to_qa_pair(doc, gen_prefix, include_answer=False)
            messages = [messages]

        if apply_chat_template and chat_template:
            res = [
                self._render_chat_template(
                    m,
                    chat_template,
                    tgt_delim=tgt_delim,
                    few_delim=few_delim,
                    multiturn=fewshot_as_multiturn,
                )
                for m in messages
            ]
        else:
            res = [
                self._message_to_text(m, tgt_delim=tgt_delim, few_delim=few_delim)
                for m in messages
            ]

        return res[0] if not self.multiple_inputs else res

    @staticmethod
    def _message_to_text(
        messages: list[Message],
        *,
        tgt_delim=" ",
        few_delim="\n\n",
    ) -> str:
        buff = []
        for i, m in enumerate(messages):
            if m.role == "system" or m.role == "user":
                buff.append(m.content)
            elif m.role == "assistant":
                buff.append(tgt_delim + m.content)
                if i != len(messages) - 1:
                    # then this is not assis prefill
                    buff.append(few_delim)

        return "".join(buff)

    def construct_requests(
        self, doc: dict[str, str], ctx: str | list[str], **kwargs
    ) -> list[Instance] | Instance:
        apply_chat_template = kwargs.pop("apply_chat_template", False)
        chat_template: Callable | None = kwargs.pop("chat_template", None)  # noqa: F841

        aux_arguments = None

        if self.OUTPUT_TYPE == "loglikelihood":
            arguments = (ctx, self.doc_to_target(doc))
        elif self.OUTPUT_TYPE == "loglikelihood_rolling":
            arguments = (self.doc_to_target(doc),)
        elif self.OUTPUT_TYPE == "multiple_choice":
            choices = self.doc_to_choice(doc)
            target_delimiter = (
                ""
                if (apply_chat_template and not self.config.gen_prefix)
                else self.config.target_delimiter
            )
            if self.multiple_inputs:
                # If there are multiple inputs, assume only one choice
                arguments = [(_ctx, f"{target_delimiter}{choices[0]}") for _ctx in ctx]
            else:
                # Otherwise they are placed in the continuation
                arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

            if "acc_mutual_info" in [m.metric_name for m in self.config._metric_list]:
                # if we are calculating multiple choice accuracy
                # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

                # here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # in other words normalizing by subtracting the unconditional logprob of each choice.
                aux_arguments = [
                    ("", f"{target_delimiter}{choice}") for choice in choices
                ]

                arguments.extend(aux_arguments)

        elif self.OUTPUT_TYPE == "generate_until":
            arguments = (ctx, deepcopy(self.config.generation_kwargs))

        multimodal_arg = {}
        if (
            self.config.doc_to_image
        ):  # TODO: ensure that non-multimodal tasks aren't getting visual args
            multimodal_arg = {
                **multimodal_arg,
                "visual": self.doc_to_image(doc),
            }

        if (
            self.config.doc_to_audio
        ):  # TODO: ensure that non-multimodal tasks aren't getting audio args
            multimodal_arg = {
                **multimodal_arg,
                "audio": self.doc_to_audio(doc),
            }

        if bool(multimodal_arg):
            if isinstance(arguments, list):
                arguments = [arg + (multimodal_arg,) for arg in arguments]
            else:
                arguments = arguments + (multimodal_arg,)

        if self.OUTPUT_TYPE == "multiple_choice":
            request_list = [
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=arg,
                    idx=i,
                    **kwargs,
                )
                for i, arg in enumerate(arguments)
            ]

            return request_list

        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=arguments,
            idx=0,
            **kwargs,
        )

    def process_results(self, doc: dict, results: list) -> dict[str, Any]:
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)
        result_dict = {}
        use_metric = list(m.metric_name for m in self.config._metric_list)
        if self.OUTPUT_TYPE == "loglikelihood":
            results = results[0]
            ll, is_greedy = results
            return {
                **({"perplexity": ll} if "perplexity" in use_metric else {}),
                **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
            }
        elif self.OUTPUT_TYPE == "loglikelihood_rolling":
            (loglikelihood, *_) = results
            assert isinstance(_target := self.doc_to_target(doc), str), (
                "Require target to be a string for loglikelihood_rolling"
            )
            _words = self.count_words(_target)
            _bytes = self.count_bytes(_target)
            return {
                **(
                    {"word_perplexity": (loglikelihood, _words)}
                    if "word_perplexity" in use_metric
                    else {}
                ),
                **(
                    {"byte_perplexity": (loglikelihood, _bytes)}
                    if "byte_perplexity" in use_metric
                    else {}
                ),
                **(
                    {"bits_per_byte": (loglikelihood, _bytes)}
                    if "bits_per_byte" in use_metric
                    else {}
                ),
            }
        elif self.OUTPUT_TYPE == "multiple_choice":
            lls, is_greedy = zip(*results)

            # retrieve choices in list[str] form, to compute choice lengths, etc.
            choices = (
                self.doc_to_choice(doc)
                if not self.multiple_inputs
                else cast(list[str], self.doc_to_text(doc))
            )
            completion_len = np.array([float(len(i)) for i in choices])

            if 2 * len(choices) == len(lls) and "acc_mutual_info" in use_metric:
                # then we are doing mutual info.
                # this stores the "dryrun" / unconditional answer loglikelihoods
                # as we extend the args list with unconditional ("", continuation) pairs
                lls_unconditional = lls[len(choices) :]
                if len(lls_unconditional) != len(choices):
                    raise ValueError
                # and this stores our "regular" conditional loglikelihoods
                lls = lls[: len(choices)]
            else:
                lls_unconditional = None

            pred = np.argmax(lls)
            pred_norm = np.argmax(lls / completion_len)

            gold = backup = self.doc_to_target(doc)

            if isinstance(gold, list):
                gold = [validate_index(g, len(choices)) for g in gold]
                gold_index_error = -100 in gold
            else:
                if isinstance(gold, int):
                    gold = validate_index(gold, len(choices))
                elif isinstance(gold, str):
                    gold = choices.index(gold) if gold in choices else -100

                gold_index_error = gold == -100

            if gold_index_error:
                eval_logger.warning(
                    f"Label [{backup}] index was not in within range of available choices {choices},"
                    f"Sample:\n\n{doc}\n\n"
                )

            if self.multiple_targets:
                acc = 1.0 if pred in gold else 0.0
                acc_norm = 1.0 if pred_norm in gold else 0.0
                exact_match = int(any(is_greedy[i] if i != -100 else 0 for i in gold))
            else:
                acc = 1.0 if pred == gold else 0.0
                acc_norm = 1.0 if pred_norm == gold else 0.0
                # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
                exact_match = int(is_greedy[gold]) if gold != -100 else 0

            prob_norm = utils.softmax(lls)

            # TODO use keyword arguments to the metric?
            # gold, pred, norm stuff, the original lls,
            result_dict = {
                **({"acc": acc} if "acc" in use_metric else {}),
                **({"f1": (gold, pred)} if "f1" in use_metric else {}),
                **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
                **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
                **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
                **(
                    {"brier_score": (gold, prob_norm)}
                    if "brier_score" in use_metric
                    else {}
                ),
            }

            if "acc_mutual_info" in use_metric:
                assert lls_unconditional is not None, (
                    "lls_unconditional should not be None if acc_mutual_info is in use_metric"
                )
                lls_mutual_info = [
                    ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional)
                ]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif self.OUTPUT_TYPE == "generate_until":
            gold = self.doc_to_target(doc)
            result = results[0]
            for metric in self.config._metric_list:
                try:
                    result_score = metric.fn(
                        references=[gold] if not isinstance(gold, list) else gold,
                        predictions=[result],
                        **metric.kwargs,
                    )
                except TypeError:  # needed for now in order to use a different interface between our own metrics and HF Evaluate metrics
                    result_score = metric.fn([gold, result])
                if isinstance(result_score, dict):
                    # This allows for multiple metrics to be returned from the same function
                    for k, v in result_score.items():
                        result_dict[k] = v
                else:
                    result_dict[metric.name] = result_score
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
                "'loglikelihood', 'loglikelihood_rolling', 'generate_until' or 'multiple_choice'",
            )

        return result_dict

    def aggregation(self) -> dict:
        return {k.name: k.aggregation_fn for k in self.config._metric_list}

    def higher_is_better(self) -> dict:
        return {k.name: k.higher_is_better for k in self.config._metric_list}

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @property
    def task_name(self) -> str | None:
        return getattr(self.config, "task", None)

    def runtime_checks(self, test_doc):
        # Test One Doc
        self.features: list[str] = list(self.task_docs.features.keys())
        self.multiple_target = 0
        self.multiple_input = 0
        test_text = self.doc_to_text(test_doc)
        test_target = self.doc_to_target(test_doc)

        if self.config.doc_to_choice is not None:
            test_choice = self.doc_to_choice(test_doc)
            if not isinstance(test_choice, list):
                eval_logger.error("doc_to_choice must return list")
            else:
                num_choice = len(test_choice)

            if isinstance(test_text, int):
                eval_logger.debug(
                    "doc_to_text returned an int. Assuming multiple inputs."
                )

            if isinstance(test_text, int):
                eval_logger.debug(
                    "doc_to_text returned an int. Assuming multiple inputs."
                )
                self.multiple_input = num_choice
        else:
            test_choice = None

        if isinstance(test_target, list):
            eval_logger.debug(
                "doc_to_target returned a list. Assuming multiple targets."
            )
            self.multiple_target = len(test_target)
        else:
            if (isinstance(test_target, int)) and (test_choice is not None):
                test_target = test_choice[test_target]
            else:
                test_target = str(test_target)

        check_choices = test_choice if test_choice is not None else [test_target]
        if self.config.doc_to_choice is not None:
            for choice in check_choices:
                choice_has_whitespace = choice[0].isspace()
                delimiter_has_whitespace = (
                    self.config.target_delimiter.rstrip()
                    != self.config.target_delimiter
                )

                if delimiter_has_whitespace and choice_has_whitespace:
                    eval_logger.debug(
                        f'Both target_delimiter "{self.config.target_delimiter}" and target choice: "{choice}" have whitespace'
                    )
                elif (not delimiter_has_whitespace) and (not choice_has_whitespace):
                    eval_logger.debug(
                        f'Both target_delimiter "{self.config.target_delimiter}" and target choice: "{choice}" do not have whitespace, ignore if the language you are evaluating on does not require/use whitespace'
                    )

    def __repr__(self):
        return (
            f"ConfigurableTask(task_name={getattr(self.config, 'task', None)},"
            f"output_type={self.OUTPUT_TYPE},"
            f"num_fewshot={getattr(self.config, 'num_fewshot', None)},"
            f"num_samples={len(self.eval_docs)})"
        )


class MultipleChoiceTask(Task):
    OUTPUT_TYPE = "loglikelihood"

    def doc_to_target(self, doc: dict) -> str:
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance]:
        # TODO: add mutual info here?
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, f" {choice}"),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]

    def process_results(self, doc: dict, results: Iterable[tuple[float, bool]]) -> dict:
        results = [
            res[0] for res in results
        ]  # only retain loglikelihoods, discard is_greedy TODO: do we need is_greedy anywhere?
        gold = doc["gold"]

        acc = 1.0 if np.argmax(results) == gold else 0.0
        completion_len = np.array([float(len(i)) for i in doc["choices"]])
        acc_norm = 1.0 if np.argmax(results / completion_len) == gold else 0.0

        return {
            "acc": acc,
            "acc_norm": acc_norm,
        }

    def higher_is_better(self) -> dict:
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self) -> dict:
        from lm_eval.api.metrics import mean

        return {
            "acc": mean,
            "acc_norm": mean,
        }


class PerplexityTask(Task):
    OUTPUT_TYPE = "loglikelihood_rolling"

    def has_training_docs(self) -> bool:
        return False

    def fewshot_examples(self, k: int, rnd) -> list:
        if k != 0:
            raise ValueError(
                "The number of fewshot examples must be 0 for perplexity tasks."
            )
        return []

    def fewshot_context(self, doc: dict, num_fewshot: int) -> Literal[""]:
        if num_fewshot != 0:
            raise ValueError(
                "The number of fewshot examples must be 0 for perplexity tasks."
            )

        return ""

    def higher_is_better(self) -> dict:
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_decontamination_query(self, doc):
        return doc

    def doc_to_text(self, doc) -> str:
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc: dict, ctx: str | None, **kwargs):
        if bool(ctx):
            raise ValueError

        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(self.doc_to_target(doc),),
            idx=0,
            **kwargs,
        )

    def process_results(self, doc: dict, results: tuple[float]) -> dict:
        (loglikelihood,) = results
        words = self.count_words(self.doc_to_target(doc))
        bytes_ = self.count_bytes(self.doc_to_target(doc))
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }

    def aggregation(self) -> dict:
        from lm_eval.api.metrics import bits_per_byte, weighted_perplexity

        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    @classmethod
    def count_bytes(cls, doc) -> int:
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc) -> int:
        """Downstream tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))
