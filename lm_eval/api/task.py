from __future__ import annotations

import abc
import ast
import logging
import random
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import datasets
import numpy as np
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.instance import Instance, OutputType
from lm_eval.api.metrics import bits_per_byte, mean, weighted_perplexity
from lm_eval.api.registry import (
    AGGREGATION_REGISTRY,
    DEFAULT_METRIC_REGISTRY,
    get_aggregation,
    get_metric,
    get_metric_aggregation,
    is_higher_better,
)
from lm_eval.api.utils import (
    Message,
    ends_with_whitespace,
    maybe_delimit,
    multiturn_to_singleturn,
    random_task_id,
    requires_delimiter,
)
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.task import TaskConfig
from lm_eval.filters import build_filter_ensemble
from lm_eval.prompts import get_prompt


if TYPE_CHECKING:
    from lm_eval.config.task import FewshotConfig

ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "generate_until",
]

eval_logger = logging.getLogger(__name__)


TaskConfig = TaskConfig


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

        self._config: TaskConfig = TaskConfig({**config}) if config else TaskConfig()

        self._filters = [build_filter_ensemble("none", [["take_first", None]])]
        self.fewshot_rnd: random.Random | None = (
            None  # purposely induce errors in case of improper usage
        )

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

    @abc.abstractmethod
    def has_training_docs(self):
        """Whether the task has a training set"""
        pass

    @abc.abstractmethod
    def has_validation_docs(self):
        """Whether the task has a validation set"""
        pass

    @abc.abstractmethod
    def has_test_docs(self):
        """Whether the task has a test set"""
        pass

    def training_docs(self) -> Iterable:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self) -> Iterable:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self) -> Iterable:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self) -> Iterable:
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            if self.config.get("num_fewshot", 0) > 0:
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

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc):
        raise NotImplementedError(
            "Override doc_to_decontamination_query with document specific decontamination query."
        )

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    # not an abstractmethod because not every language-only task has to implement this
    def doc_to_image(self, doc):
        raise NotImplementedError

    def doc_to_audio(self, doc):
        raise NotImplementedError

    def doc_to_prefix(self, doc):
        return ""

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
                gen_prefix=self.doc_to_prefix(doc),
            )

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
            inst = self.construct_requests(
                doc=doc,
                ctx=fewshot_ctx,
                metadata=(self.config["task"], doc_id, self.config.repeats),
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
    def construct_requests(self, doc, ctx, **kwargs):
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
        pass

    @abc.abstractmethod
    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        pass

    @abc.abstractmethod
    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        pass

    @abc.abstractmethod
    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        pass

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @classmethod
    def count_bytes(cls, doc):
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
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

        description = description or ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs()
                        if self.has_validation_docs()
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
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (num_fewshot)
        return self.config.to_dict()

    def set_config(self, key: str, value: Any, update: bool = False) -> None:
        """Set or update the configuration for a given key."""
        if key is None:
            raise ValueError("Key must be provided.")

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
        (
            self._metric_fn_list,
            self._aggregation_list,
            self._metric_fn_kwargs,
            self._higher_is_better,
        ) = ({}, {}, {}, {})
        self._metric_fn_list[metric_name] = get_metric(metric_name)
        self._aggregation_list[metric_name] = get_metric_aggregation(metric_name)
        self._higher_is_better[metric_name] = is_higher_better(metric_name)
        self._metric_fn_kwargs[metric_name] = {}
        if not isinstance(self, ConfigurableTask):
            self.process_results = lambda x, y: {metric_name: get_metric(metric_name)}
            self.aggregation = lambda: {
                metric_name: get_metric_aggregation(metric_name)
            }
        self._config["metric_list"] = [{"metric": metric_name}]
        self._config["process_results"] = "process_results"

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        self.fewshot_rnd = random.Random(seed)
        if hasattr(self, "sampler"):
            self.sampler.set_rnd(seed)

    @property
    def eval_docs(self) -> datasets.Dataset | list[dict]:
        if self.has_test_docs():
            return self.test_docs()
        elif self.has_validation_docs():
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

    @staticmethod
    def resolve_field(doc: dict[str, Any], field: str | None = None):
        if field:
            return doc[field] if field in doc else utils.apply_template(field, doc)

    @property
    def task_name(self) -> str:
        return getattr(self.config, "task", None) or random_task_id()


class ConfigurableTask(Task):
    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config: dict | None = None,
    ) -> None:  # TODO no super() call here
        # Get pre-configured attributes
        self._config = self.CONFIG

        # Use new configurations if there was no preconfiguration
        if self.config is None:
            self._config = TaskConfig(**config)
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

        self.fewshot_cfg = cast("FewshotConfig", self.config.fewshot_config)

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

        self._metric_fn_list = {}
        self._metric_fn_kwargs = {}
        self._aggregation_list = {}
        self._higher_is_better = {}

        if self.config.metric_list is None:
            # TODO: handle this in TaskConfig.__post_init__ ?
            _metric_list = DEFAULT_METRIC_REGISTRY[self.config.output_type]

            for metric_name in _metric_list:
                self._metric_fn_list[metric_name] = get_metric(metric_name)
                self._metric_fn_kwargs[metric_name] = {}
                self._aggregation_list[metric_name] = get_metric_aggregation(
                    metric_name
                )
                self._higher_is_better[metric_name] = is_higher_better(metric_name)
        else:
            for metric_config in self.config.metric_list:
                if "metric" not in metric_config:
                    raise ValueError(
                        "'metric' key not provided for an entry in 'metric_list', must be specified!"
                    )
                metric_name = metric_config["metric"]
                kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key
                    not in ["metric", "aggregation", "higher_is_better", "hf_evaluate"]
                }
                hf_evaluate_metric = (
                    "hf_evaluate" in metric_config
                    and metric_config["hf_evaluate"] is True
                )

                if self.config.process_results is not None:
                    self._metric_fn_list[metric_name] = None
                    self._metric_fn_kwargs[metric_name] = {}
                elif callable(metric_name):
                    metric_fn = metric_name.__call__
                    metric_name = metric_name.__name__
                    self._metric_fn_list[metric_name] = metric_fn
                    self._metric_fn_kwargs[metric_name] = kwargs
                else:
                    self._metric_fn_list[metric_name] = get_metric(
                        metric_name, hf_evaluate_metric
                    )
                    self._metric_fn_kwargs[metric_name] = kwargs

                if "aggregation" in metric_config:
                    agg_name = metric_config["aggregation"]
                    if isinstance(agg_name, str):
                        self._aggregation_list[metric_name] = get_aggregation(agg_name)
                    elif callable(agg_name):  # noqa: E721
                        self._aggregation_list[metric_name] = metric_config[
                            "aggregation"
                        ]
                else:
                    INV_AGG_REGISTRY = {v: k for k, v in AGGREGATION_REGISTRY.items()}
                    metric_agg = get_metric_aggregation(metric_name)
                    eval_logger.warning(
                        f"[Task: {self.config.task}] metric {metric_name} is defined, but aggregation is not. "
                        f"using default "
                        f"aggregation={INV_AGG_REGISTRY[metric_agg]}"
                    )
                    self._aggregation_list[metric_name] = metric_agg

                if "higher_is_better" in metric_config:
                    self._higher_is_better[metric_name] = metric_config[
                        "higher_is_better"
                    ]
                else:
                    eval_logger.warning(
                        f"[Task: {self.config.task}] metric {metric_name} is defined, but higher_is_better is not. "
                        f"using default "
                        f"higher_is_better={is_higher_better(metric_name)}"
                    )
                    self._higher_is_better[metric_name] = is_higher_better(metric_name)

        self.download(self.config.dataset_kwargs)
        self._training_docs = None
        self._fewshot_docs = None

        if self.config.filter_list is not None:
            self._filters = []
            for filter_config in self.config.filter_list:
                filter_name = filter_config["name"]
                filter_functions = filter_config["filter"]
                components = []
                for function in filter_functions:
                    kwargs = {
                        key: function[key] for key in function if key != "function"
                    }
                    components.append([function["function"], kwargs])
                filter_pipeline = build_filter_ensemble(filter_name, components)
                self._filters.append(filter_pipeline)
        else:
            # TODO: handle repeats in a more general way rather than just discarding
            if self.OUTPUT_TYPE == "generate_until":
                eval_logger.debug(
                    "No custom filters defined. Using default 'take_first' filter for handling repeats."
                )
            self._filters = [build_filter_ensemble("none", [["take_first", None]])]

        if self.config.use_prompt is not None:
            eval_logger.info(f"loading prompt {self.config.use_prompt}")
            self.prompt = get_prompt(
                self.config.use_prompt, self.DATASET_PATH, self.DATASET_NAME
            )
        else:
            self.prompt = None

        if (_fs_docs := self.fewshot_docs()) is not None:
            config_sampler: str | type[samplers.ContextSampler] = (
                self.fewshot_cfg.sampler if self.config.fewshot_config else "default"
            )
            fewshot_docs = list(_fs_docs)  # type: ignore
            if isinstance(config_sampler, str):
                sampler_cls = samplers.get_sampler(config_sampler)
            elif issubclass(config_sampler, samplers.ContextSampler):
                sampler_cls = config_sampler
            else:
                raise TypeError(
                    f"fewshot_config.sampler should be a string or subclass of ContextSampler, "
                    f"not {type(config_sampler)}"
                )
            self.sampler: samplers.ContextSampler = sampler_cls(fewshot_docs, rnd=None)  # type: ignore

        self.task_docs = self.eval_docs

        # Test One Doc
        self.features = list(self.task_docs.features.keys())
        self.multiple_input = 0
        self.multiple_target = 0
        test_doc = self.task_docs[0]
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
                choice_has_whitespace = bool(choice[0].isspace())
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

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        from packaging.version import parse as vparse

        if dataset_kwargs and vparse(datasets.__version__) >= vparse("4.0.0"):
            dataset_kwargs.pop("trust_remote_code", None)
        if isinstance(self.config.custom_dataset, Callable):
            eval_logger.warning(
                f"{self.config.task}: Custom kwargs can be passed to `--metadata` in console (as json string) or to the TaskManager."
                + "\nFor example --metadata='{\"max_seq_lengths\":[4096, 8192]}'. For details see task Readme."
            )
            self.dataset = self.config.custom_dataset(
                **(self.config.metadata or {}), **(self.config.dataset_kwargs or {})
            )
        else:
            self.dataset = datasets.load_dataset(
                path=self.DATASET_PATH,
                name=self.DATASET_NAME,
                **dataset_kwargs if dataset_kwargs is not None else {},
            )

    def has_training_docs(self) -> bool:
        return self.config.training_split is not None

    def has_validation_docs(self) -> bool:
        return self.config.validation_split is not None

    def has_test_docs(self) -> bool:
        return self.config.test_split is not None

    def training_docs(self) -> datasets.Dataset:
        if self.has_training_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.training_split]
                )
            return self.dataset[self.config.training_split]

    def validation_docs(self) -> datasets.Dataset:
        if self.has_validation_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.validation_split]
                )
            return self.dataset[self.config.validation_split]

    def test_docs(self) -> datasets.Dataset:
        if self.has_test_docs():
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.test_split])
            return self.dataset[self.config.test_split]

    def fewshot_docs(self):
        if (split := self.fewshot_cfg.split) is not None:
            if (process_docs := self.fewshot_cfg.process_docs) is not None:
                return process_docs(self.dataset[split])
            return self.dataset[split]
        elif (
            self.config.fewshot_config is not None
            and (samples := self.fewshot_cfg.samples) is not None
        ):
            # fmt: off
            match samples:
                case list(): return samples
                case fsamples if callable(samples): return fsamples()
                case _: raise Exception(
                        "`fewshot_config['samples']` was incorrectly defined in the configuration. It should either be `list[dict]`, or callable returning this list."
                    ) from None
            # fmt: on
        else:
            if (_shots := self.config.num_fewshot) is not None and (_shots > 0):
                eval_logger.warning(
                    f"[Task: {self.config.task}] "
                    f"num_fewshot > 0 but fewshot_split is None. "
                    "using preconfigured rule."
                )
            return super().fewshot_docs()

    @utils.positional_deprecated
    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable[..., str] | None = None,
        gen_prefix: str | None = None,
    ) -> str | list[str]:
        """Build the full prompt context including system prompt, few-shot examples, and eval doc.

        Constructs a complete prompt by:
        1. Adding system instruction + task description (if provided)
        2. Adding `num_fewshot` labeled examples from the fewshot split
        3. Adding the evaluation document (without its answer)

        Each component is built using `build_qa_turn()` and can be rendered as plain
        text or formatted via a chat template.

        Args:
            doc (dict): The evaluation document to build context for.
            num_fewshot (int): Number of few-shot examples to include.
            system_instruction (str | None): System instruction to prepend to the prompt.
            apply_chat_template (bool): If True, format output using the chat template.
            fewshot_as_multiturn (bool): If True, keep few-shot examples as separate
                user/assistant turns. If False, collapse into a single user message.
            chat_template (Callable | None): Renders a list of message dicts to a string.
            gen_prefix (str | None): Prefix to start the assistant's response (e.g., "Answer:").

        Returns:
            str | list[str]: The formatted prompt string, or a list of strings for
                multiple-input tasks (e.g., Winogrande where each choice becomes a
                separate context).
        """
        messages = []
        chat_template = (
            partial(chat_template, add_generation_prompt=not gen_prefix)
            if chat_template
            else None
        )
        description = self.resolve_field(doc, self.config.description) or ""
        system_prompt = maybe_delimit(
            system_instruction, description, self.config.fewshot_delimiter
        )
        if system_prompt:
            messages.append(Message("system", system_prompt))

        if num_fewshot > 0:
            for fs_doc in self.sampler.sample(
                n=num_fewshot,
                eval_doc=doc
                if self.fewshot_cfg.split == self.config.test_split
                else None,
            ):
                q, c, a = (
                    self.doc_to_text(fs_doc, self.fewshot_cfg.doc_to_text),
                    self.doc_to_choice(fs_doc, self.fewshot_cfg.doc_to_choice)
                    if self.fewshot_cfg.doc_to_choice
                    else None,
                    self.doc_to_target(fs_doc, self.fewshot_cfg.doc_to_target),
                )
                _gen_prefix = self.resolve_field(doc, self.fewshot_cfg.gen_prefix)
                # for multiple inputs, q: int, c: list[str], target: str
                # TODO: fix this hacky way of handling multiple inputs
                if self.multiple_input:
                    q = cast("str", c[q])  # type: ignore
                    c = None
                # TODO: fix types
                messages += self.build_qa_turn(
                    q=q,
                    c=c,
                    a=a,
                    gen_prefix=_gen_prefix,
                    tgt_delim=self.fewshot_cfg.target_delimiter,
                    few_delim=self.fewshot_cfg.fewshot_delimiter,
                )

        q, c, a = (
            self.doc_to_text(doc),
            self.doc_to_choice(doc) if self.config.doc_to_choice else None,
            self.doc_to_target(doc),
        )
        if self.multiple_input:
            assert isinstance(c, list), "multiple inputs require choices to be a list"
            return self.multiple_input_context(
                messages,
                gen_prefix,
                c,
                chat_template=chat_template if apply_chat_template else None,
                fewshot_as_multiturn=fewshot_as_multiturn,
            )
        messages += self.build_qa_turn(
            q=q,
            c=c,
            gen_prefix=gen_prefix,
            tgt_delim=self.config.target_delimiter,
            few_delim="",
        )
        if apply_chat_template and chat_template:
            res = (
                [m.to_dict() for m in messages]
                if fewshot_as_multiturn
                else multiturn_to_singleturn(messages)
            )
            res = chat_template(res)
        else:
            res = "".join(m.to_text() for m in messages)

        return res

    def build_qa_turn(
        self,
        *,
        q: str | None = None,
        c: list[str] | None = None,
        a: str | int | list[str] | None = None,
        gen_prefix: str | None = None,
        tgt_delim=" ",
        few_delim="\n\n",
    ) -> list[Message]:
        r"""Build a single Q&A turn as a list of Messages.

        Constructs a user message containing the question/context, and optionally
        an assistant message containing the answer. Used for building both few-shot
        examples and the final evaluation prompt. The returned Messages can be
        rendered as plain text (via to_text()) or converted to chat format
        (via to_dict()) depending on whether a chat template is applied.

        Args:
            q (str): The question or context text (required).
            c (list[str] | None): List of answer choices for multiple-choice tasks.
                When provided with an integer `a`, indexes into this list to get the answer.
            a (str | int | list[str] | None): The answer - can be a string, an index
                into `c`, or a list of strings (for multiple targets).
            gen_prefix (str | None): A prefix to prepend to generated text (e.g., "Answer:").
            tgt_delim (str): Delimiter between question and answer (default: " ").
            few_delim (str): Delimiter after assistant response for few-shot separation
                (default: "\n\n").

        Returns:
            list[Message]: [user_msg] or [user_msg, assistant_msg] depending on
                whether an answer or gen_prefix is provided.
        """
        assert isinstance(q, str), f"Context is not a string! : {q}"
        # Check if answer is provided (handle a=0 as valid answer index)
        has_answer = a is not None and a != ""
        msgs = [
            Message(
                "user",
                q,
                tgt_delim
                if has_answer and not gen_prefix
                else tgt_delim
                if gen_prefix and requires_delimiter(q, gen_prefix)
                else "",
            )
        ]
        if has_answer:
            answer_text = (
                c[a]
                if (c and isinstance(a, int))
                # TODO: for multiple targets a is a list[str]. Fix this hack
                else a[0]
                if isinstance(a, list)
                else a
            )
            assert isinstance(answer_text, str), f"Answer is not a string! : {a}"
            # Currently, we always delimit gen_prefex and answer with space if deliimter required.
            answer_text = maybe_delimit(gen_prefix, answer_text, delimiter=" ")
            msgs.append(Message("assistant", answer_text, few_delim))
        elif gen_prefix:
            # For gen-prefix, the delimiter is added in construct_requests
            msgs.append(Message("assistant", gen_prefix))
        return msgs

    def multiple_input_context(
        self,
        prev_context: list[Message] | None,
        gen_prefix: str | None,
        q: list[str],
        chat_template: Callable[..., str] | None = None,
        fewshot_as_multiturn: bool = False,
    ) -> list[str]:
        """Build separate prompt contexts for each input choice in multiple-input tasks.

        For tasks like Winogrande where each answer choice produces a different
        input context (e.g., filling a blank with different options), this method
        creates a separate full prompt for each choice. All prompts share the same
        fewshot prefix but differ in the final evaluation turn.

        Args:
            prev_context (list[Message] | None): Messages from system prompt and fewshot
                examples (shared across all choices).
            gen_prefix (str | None): Prefix to start the assistant's response (e.g., "Answer:").
            q (list[str]): List of input texts, one per choice.
            chat_template (Callable | None): Renders a list of message dicts to a string.
            fewshot_as_multiturn (bool): If True, keep messages as separate turns.

        Returns:
            list[str]: Formatted prompt strings, one per input choice.
        """
        # for multiple inputs, q is list[str]
        res_ = []
        prev_context = prev_context or []
        contexts = [
            prev_context
            + self.build_qa_turn(
                q=ctx,
                gen_prefix=gen_prefix,
                tgt_delim="",
            )
            for ctx in q
        ]
        for messages in contexts:
            if chat_template:
                res = (
                    [m.to_dict() for m in messages]
                    if fewshot_as_multiturn
                    else multiturn_to_singleturn(messages)
                )
                res = chat_template(res)
            else:
                res = "".join(m.to_text() for m in messages)
            res_.append(res)
        return res_

    def apply_filters(self) -> list[Instance] | None:
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances)
        else:
            eval_logger.warning("No filter defined, passing through instances")
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

    def doc_to_text(self, doc, doc_to_text=None):
        if self.prompt is not None:
            doc_to_text = self.prompt
        elif doc_to_text is not None:
            doc_to_text = doc_to_text
        else:
            doc_to_text = self.config.doc_to_text

        if isinstance(doc_to_text, int):
            return doc_to_text
        elif isinstance(doc_to_text, str):
            if doc_to_text in self.features:
                # if self.config.doc_to_choice is not None:
                #     return self.doc_to_choice(doc)[doc[doc_to_text]]
                # else:
                return doc[doc_to_text]
            else:
                text_string = utils.apply_template(doc_to_text, doc)
                if text_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(text_string)
                else:
                    return text_string
        elif callable(doc_to_text):
            return doc_to_text(doc)
        # Used when applying a Promptsource template
        elif hasattr(doc_to_text, "apply"):
            applied_prompt = doc_to_text.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[0]
            else:
                eval_logger.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            print(type(doc_to_text))
            raise TypeError

    def doc_to_target(self, doc: Mapping, doc_to_target=None) -> int | str | list:
        if self.prompt is not None:
            doc_to_target = self.prompt
        elif doc_to_target is not None:
            doc_to_target = doc_to_target
        else:
            doc_to_target = self.config.doc_to_target

        if isinstance(doc_to_target, int):
            return doc_to_target
        elif isinstance(doc_to_target, str):
            if doc_to_target in self.features:
                # if self.config.doc_to_choice is not None:
                #     return self.doc_to_choice(doc)[doc[doc_to_target]]
                # else:
                return doc[doc_to_target]
            else:
                target_string = utils.apply_template(doc_to_target, doc)
                if target_string.isdigit() and self._config.doc_to_choice is not None:
                    return ast.literal_eval(target_string)
                elif (
                    len(target_string) >= 2
                    and (target_string[0] == "[")
                    and (target_string[-1] == "]")
                ):
                    try:
                        return ast.literal_eval(target_string)
                    except (SyntaxError, ValueError):
                        return target_string
                else:
                    return target_string
        elif isinstance(doc_to_target, list):
            return doc_to_target
        elif callable(doc_to_target):
            return doc_to_target(doc)
        # Used when applying a Promptsource template
        elif hasattr(doc_to_target, "apply"):
            applied_prompt = doc_to_target.apply(doc)
            if len(applied_prompt) == 2:
                return applied_prompt[1]
            else:
                eval_logger.warning("Applied prompt returns empty string")
                return self.config.fewshot_delimiter
        else:
            raise TypeError

    def doc_to_choice(self, doc: Any, doc_to_choice=None) -> list[str]:
        if self.prompt is not None:
            doc_to_choice = self.prompt
        elif doc_to_choice is not None:
            doc_to_choice = doc_to_choice
        elif self.config.doc_to_choice is None:
            eval_logger.error("doc_to_choice was called but not set in config")
        else:
            doc_to_choice = self.config.doc_to_choice

        if isinstance(doc_to_choice, str):
            if doc_to_choice in self.features:
                return doc[doc_to_choice]
            else:
                return ast.literal_eval(utils.apply_template(doc_to_choice, doc))
        elif isinstance(doc_to_choice, list):
            return doc_to_choice
        elif isinstance(doc_to_choice, dict):
            return list(doc_to_choice.values())
        elif callable(doc_to_choice):
            return doc_to_choice(doc)
        elif hasattr(doc_to_choice, "get_answer_choices_list"):
            return doc_to_choice.get_answer_choices_list(doc)
        else:
            raise TypeError

    def doc_to_image(self, doc: Any, doc_to_image=None) -> int | str | list | None:
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

    def doc_to_prefix(self, doc):
        if (gen_prefix := self.config.gen_prefix) is not None:
            if gen_prefix in self.features:
                return doc[gen_prefix]
            else:
                return utils.apply_template(gen_prefix, doc)
        return None

    def construct_requests(
        self, doc: dict, ctx: str | list[str], **kwargs
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
            target_delimiter = self.config.target_delimiter
            if apply_chat_template:
                target_delimiter = (
                    self.config.target_delimiter
                    if self.config.gen_prefix
                    and not ends_with_whitespace(self.config.gen_prefix)
                    else ""
                )
            if self.multiple_input:
                # If there are multiple inputs, choices are placed in the ctx
                cont = self.doc_to_target(doc)
                arguments = [(context, f"{target_delimiter}{cont}") for context in ctx]
            else:
                # Otherwise they are placed in the continuation
                arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

            # TODO: we should raise a warning telling users this will at most ~2x runtime.
            if "acc_mutual_info" in self._metric_fn_list.keys():
                # if we are calculating multiple choice accuracy
                # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

                # here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # in other words normalizing by subtracting the unconditional logprob of each choice.
                # TODO: should these be strided? will have to modify the processing in process_results if so
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
                **{"visual": self.doc_to_image(doc)},
            }

        if (
            self.config.doc_to_audio
        ):  # TODO: ensure that non-multimodal tasks aren't getting audio args
            multimodal_arg = {
                **multimodal_arg,
                **{"audio": self.doc_to_audio(doc)},
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

    def process_results(self, doc, results):
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)

        result_dict = {}
        use_metric = list(self._metric_fn_list.keys())
        if self.OUTPUT_TYPE == "loglikelihood":
            results = results[0]
            ll, is_greedy = results
            return {
                **({"perplexity": ll} if "perplexity" in use_metric else {}),
                **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
            }
        elif self.OUTPUT_TYPE == "loglikelihood_rolling":
            (loglikelihood,) = results
            _words = self.count_words(self.doc_to_target(doc))
            _bytes = self.count_bytes(self.doc_to_target(doc))
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
            lls, is_greedy = zip(*results, strict=True)

            # retrieve choices in List[str] form, to compute choice lengths, etc.
            choices = self.doc_to_choice(doc)
            completion_len = np.array([float(len(i)) for i in choices])
            byte_length = np.array([float(len(i.encode("utf-8"))) for i in choices])

            if (
                2 * len(choices) == len(lls)
                and "acc_mutual_info" in self._metric_fn_list.keys()
            ):
                # then we are doing mutual info.
                # this stores the "dryrun" / unconditional answer loglikelihoods
                # as we extend the args list with unconditional ("", continuation) pairs
                lls_unconditional = lls[len(choices) :]
                if len(lls_unconditional) != len(choices):
                    raise ValueError
                # and this stores our "regular" conditional loglikelihoods
                lls = lls[: len(choices)]

            pred = np.argmax(lls)
            pred_norm = np.argmax(lls / completion_len)
            pred_byte = np.argmax(lls / byte_length)

            if self.multiple_input:
                gold = self.doc_to_text(doc)
            else:
                gold = self.doc_to_target(doc)

            gold_index_error = False
            if isinstance(gold, list):
                gold = [i if i < len(choices) else -100 for i in gold]
                if -100 in gold:
                    gold_index_error = True
            else:
                if isinstance(gold, int):
                    gold = gold if gold < len(choices) else -100
                elif isinstance(gold, str):
                    gold = choices.index(gold) if gold in choices else -100

                if gold == -100:
                    gold_index_error = True

            if gold_index_error:
                eval_logger.warning(
                    f"Label index was not in within range of available choices,"
                    f"Sample:\n\n{doc}\n\n"
                )

            if self.multiple_target:
                acc = 1.0 if pred in gold else 0.0
                acc_norm = 1.0 if pred_norm in gold else 0.0
                acc_bytes = 1.0 if pred_byte in gold else 0.0
                exact_match = int(any(is_greedy[i] if i != -100 else 0 for i in gold))
            else:
                acc = 1.0 if pred == gold else 0.0
                acc_norm = 1.0 if pred_norm == gold else 0.0
                acc_bytes = 1.0 if pred_byte == gold else 0.0
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
                **({"acc_bytes": acc_bytes} if "acc_bytes" in use_metric else {}),
                **({"exact_match": exact_match} if "exact_match" in use_metric else {}),
                **(
                    {"brier_score": (gold, prob_norm)}
                    if "brier_score" in use_metric
                    else {}
                ),
                **({"likelihood": (gold, lls)} if "likelihood" in use_metric else {}),
            }

            if "acc_mutual_info" in use_metric:
                lls_mutual_info = [
                    ll_c - ll_u
                    for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
                ]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif self.OUTPUT_TYPE == "generate_until":
            gold = self.doc_to_target(doc)
            result = results[0]
            if self.config.doc_to_choice is not None:
                # If you set doc_to_choice,
                # it assumes that doc_to_target returns a number.
                choices = self.doc_to_choice(doc)
                gold = choices[gold]
            # we expect multiple_targets to be a list.
            elif self.multiple_target:
                gold = list(gold)
            # TODO: handle this better
            elif type(gold) is not type(result) and not (
                "bypass" in self._metric_fn_list.keys() or isinstance(result, list)
            ):
                # cast gold to the same type as result
                gold = type(result)(gold)

            for metric in self._metric_fn_list.keys():
                if self.multiple_target:
                    # in the case where we have multiple targets,
                    # return true if any are true
                    # TODO: this may break for multipLe_target, non zero-or-1 metrics
                    scores = []
                    if not isinstance(gold, list):
                        # sometimes, a multiple_target dataset has exceptions where one doc has only one string answer
                        # print(gold)
                        gold = [gold]
                    if metric == "exact_match":
                        result = [result for _ in range(len(gold))]
                        scores = self._metric_fn_list[metric](
                            references=gold,
                            predictions=result,
                            **self._metric_fn_kwargs[metric],
                        )[metric]
                        result_score = 1.0 if scores > 0.0 else 0.0
                    else:
                        for gold_option in gold:
                            try:
                                result_score = self._metric_fn_list[metric](
                                    references=[gold_option],
                                    predictions=[result],
                                    **self._metric_fn_kwargs[metric],
                                )
                            except (
                                TypeError
                            ):  # TODO: this is hacky and I don't want to do it
                                result_score = self._metric_fn_list[metric](
                                    [gold_option, result]
                                )
                            if isinstance(result_score, dict):
                                # TODO: this handles the case where HF evaluate returns a dict.
                                result_score = result_score[metric]
                            scores.append(result_score)
                        result_score = 1.0 if any(scores) else 0.0
                else:
                    try:
                        result_score = self._metric_fn_list[metric](
                            references=[gold],
                            predictions=[result],
                            **self._metric_fn_kwargs[metric],
                        )
                    except TypeError:  # needed for now in order to use a different interface between our own metrics and HF Evaluate metrics
                        result_score = self._metric_fn_list[metric]([gold, result])
                if isinstance(result_score, dict):
                    # TODO: this handles the case where HF evaluate returns a dict.
                    # This allows for multiple metrics to be returned from the same function
                    for k, v in result_score.items():
                        result_dict[k] = v
                else:
                    result_dict[metric] = result_score
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
                "'loglikelihood', 'loglikelihood_rolling', 'generate_until' or 'multiple_choice'",
            )

        return result_dict

    def aggregation(self) -> dict:
        return self._aggregation_list

    def higher_is_better(self) -> dict:
        return self._higher_is_better

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @property
    def task_name(self) -> str:
        return getattr(self.config, "task", random_task_id())

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
