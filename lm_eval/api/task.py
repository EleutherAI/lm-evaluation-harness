import abc
from dataclasses import dataclass, field, asdict

import re
import ast
import yaml
import evaluate
import random
import itertools
import functools

import datasets
import numpy as np

from typing import Union
from collections.abc import Callable

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.instance import Instance
from lm_eval.api.filter import FilterEnsemble

from lm_eval.logger import eval_logger
from lm_eval.prompts import get_prompt
from lm_eval.filters import build_filter_ensemble
from lm_eval.api.metrics import (
    # get_metric,
    # get_aggregation,
    mean,
    weighted_perplexity,
    bits_per_byte,
)
from lm_eval.api.registry import (
    METRIC_REGISTRY,
    DEFAULT_METRIC_REGISTRY,
    OUTPUT_TYPE_REGISTRY,
    AGGREGATION_REGISTRY,
    HIGHER_IS_BETTER_REGISTRY,
    DEFAULT_AGGREGATION_REGISTRY,
)

ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "greedy_until",
]


@dataclass
class TaskConfig(dict):

    task: str = None
    group: Union[str, list] = None
    reference: str = None

    dataset_path: str = None
    dataset_name: str = None
    dataset_kwargs: dict = None
    training_split: str = None
    validation_split: str = None
    test_split: str = None
    fewshot_split: str = None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaling (?)

    template_aliases: str = None
    aliases: Union[str, list] = None
    doc_to_text: Union[Callable, str] = None
    doc_to_target: Union[Callable, str] = None
    use_prompt: str = None

    num_fewshot: int = 0
    batch_size: int = 1
    repeats: int = 1

    metric_list: str = None
    gold_alias: Union[Callable, str] = None
    output_type: str = "greedy_until"
    generation_kwargs: dict = None
    delimiter: str = "\n\n"
    filter_list: Union[str, list] = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str = None

    metadata: str = None  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    def __post_init__(self):
        # allow user-specified aliases so that users can
        # force prompt-compatibility for some prompt regardless of
        # field names in prompt
        if self.template_aliases is not None:
            if type(self.doc_to_text) == str:
                self.doc_to_text = self.template_aliases + self.doc_to_text

            if type(self.doc_to_target) == str:
                self.doc_to_target = self.template_aliases + self.doc_to_target

            if type(self.gold_alias) == str:
                self.gold_alias = self.template_aliases + self.gold_alias

        if self.generation_kwargs:
            assert (
                self.output_type == "greedy_until"
            ), "passed `generation_kwargs`, but not using a generation request type!"
        elif self.output_type == "greedy_until":
            # ensure that we greedily generate in absence of explicit arguments otherwise
            self.generation_kwargs = {"do_sample": False, "temperature": 0.0}

        # TODO: how to make TaskConfigs be de- and re-serializable, even when using the !function constructor?

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self):
        """dumps the current config as a dictionary object, as a printable format.
        null fields will not be printed.
        Used for dumping results alongside full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        # TODO: should any default value in the TaskConfig not be printed?
        """
        cfg_dict = asdict(self)
        # remove values that are `None`
        for k, v in list(cfg_dict.items()):
            if v is None:
                cfg_dict.pop(k)
            elif isinstance(v, Callable):
                # TODO: this should handle Promptsource template objects as a separate case?
                cfg_dict[k] = str(v)
        return cfg_dict


class Task(abc.ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object which represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...} or
        {"question": ..., question, answer)
    """

    VERSION = None

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    # or a path to a custom `datasets` loading script.
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    OUTPUT_TYPE: str = None

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        download_mode=None,
        config=None,
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
        self._instances = None

        self._config = TaskConfig(**config) if config else TaskConfig()

        if not hasattr(self, "_filters"):
            self._filters = []
            for name, components in self._config.get(
                "filters", [["none", [["take_first", None]]]]
            ):
                filter_pipeline = build_filter_ensemble(name, components)
                self._filters.append(filter_pipeline)

        self.sampler = samplers.Sampler(
            list(self.fewshot_docs()), self, rnd=random.Random()
        )  # TODO: pass the correct docs in here

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
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

    def training_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def validation_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def test_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        return []

    def fewshot_docs(self):
        """
        :return: Iterable[obj]
            A iterable of any object, that doc_to_text can handle
        """
        if self.has_training_docs():
            return self.training_docs()
        elif self.has_validation_docs():
            return self.validation_docs()
        else:
            eval_logger.warning(
                "has_training_docs and has_validation_docs are False"
                ", using test_docs but this is not recommended."
            )
            return self.test_docs()

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    @property
    def instances(self):
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    def fewshot_examples(self, k, rnd):
        if self._training_docs is None:
            self._training_docs = list(self.training_docs())

        return rnd.sample(self._training_docs, k)

    def doc_to_decontamination_query(self, doc):
        print(
            "Override doc_to_decontamination_query with document specific decontamination query."
        )
        assert False

    @abc.abstractmethod
    def doc_to_text(self, doc):
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc):
        pass

    def build_all_requests(self, limit=None, rank=None, world_size=None):
        """Build a set of Instances for a task, and store them in task.instances"""
        if self.has_test_docs():
            docs = self.test_docs()
        elif self.has_validation_docs():
            docs = self.validation_docs()
        else:
            assert (
                False
            ), f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"

        instances = []
        for doc_id, doc in utils.create_iterator(
            enumerate(docs), rank, world_size, limit
        ):
            # sample fewshot context #TODO: need to offset doc_id by rank now!
            fewshot_ctx = self.fewshot_context(
                doc, self._config.num_fewshot, rnd=random.Random()
            )
            # TODO: hardcoded for now: # of runs on each input to be 2. # TODO: we should override this if doing greedy gen so users don't waste time+compute
            inst = self.construct_requests(
                doc=doc,
                ctx=fewshot_ctx,
                metadata=(self._config["task"], doc_id, self._config.repeats),
            )

            if not isinstance(inst, list):
                inst = [inst]

            instances.extend(inst)

        self._instances = instances
        assert len(self._instances) != 0, "task.build_requests() did not find any docs!"

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

    @classmethod
    def count_bytes(cls, doc):
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """Downstream loglikelihood_rolling perplexity tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))

    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, rnd=None):
        """Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :returns: str
            The fewshot context.
        """
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`"

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            labeled_examples = self.sampler.get_context(doc, num_fewshot)

            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            # if self.has_training_docs():
            #     fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd)
            # else:
            #     if self._fewshot_docs is None:
            #         self._fewshot_docs = list(
            #             self.validation_docs()
            #             if self.has_validation_docs()
            #             else self.test_docs()
            #         )

            #     fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

            #     # get rid of the doc that's the one we're evaluating, if it's in the fewshot
            #     fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            # labeled_examples = (
            #     "\n\n".join(
            #         [
            #             self.doc_to_text(doc) + self.doc_to_target(doc)
            #             for doc in fewshotex
            #         ]
            #     )
            #     + "\n\n"
            # )

        example = self.doc_to_text(doc)
        return labeled_examples + example

    def apply_filters(self):

        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def dump_config(self):
        """Returns a dictionary representing the task's config.

        :returns: str
            The fewshot context.
        """
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (batch size, num_fewshot)
        return self._config.to_dict()


class ConfigurableTask(Task):

    VERSION = "Yaml"
    OUTPUT_TYPE = None
    CONFIG = None

    def __init__(
        self, data_dir=None, cache_dir=None, download_mode=None, config: dict = None
    ):
        # Get pre-configured attributes
        self._config = self.CONFIG

        # Use new configurations if there was no preconfiguration
        if self._config is None:
            self._config = TaskConfig(**config)
        # Overwrite configs
        else:
            if config is not None:
                self._config.__dict__.update(config)

        if self._config is None:
            raise ValueError(
                "Must pass a config to ConfigurableTask, either in cls.CONFIG or `config` kwarg"
            )

        if self._config.output_type is not None:
            assert self._config.output_type in ALL_OUTPUT_TYPES
            self.OUTPUT_TYPE = self._config.output_type

        if self._config.dataset_path is not None:
            self.DATASET_PATH = self._config.dataset_path

        if self._config.dataset_name is not None:
            self.DATASET_NAME = self._config.dataset_name

        self._metric_fn_list = {}
        self._metric_fn_kwargs = {}
        self._aggregation_list = {}
        self._higher_is_better = {}

        _metric_list = DEFAULT_METRIC_REGISTRY[self._config.output_type]
        if self._config.metric_list is None:
            # TODO: handle this in TaskConfig.__post_init__ ?
            for metric_name in _metric_list:
                self._metric_fn_list[metric_name] = METRIC_REGISTRY[metric_name]
                self._aggregation_list[metric_name] = DEFAULT_AGGREGATION_REGISTRY[
                    metric_name
                ]
                self._higher_is_better[metric_name] = HIGHER_IS_BETTER_REGISTRY[
                    metric_name
                ]
        else:
            for metric_config in self._config.metric_list:
                assert "metric" in metric_config
                metric_name = metric_config["metric"]
                kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key not in ["metric", "aggregation", "higher_is_better"]
                }
                try:
                    self._metric_fn_list[metric_name] = METRIC_REGISTRY[metric_name]
                except Exception:
                    eval_logger.warning(
                        f"Metric {metric_name} not found, "
                        "Searching from https://huggingface.co/evaluate-metric"
                    )
                    try:
                        metric_object = evaluate.load(metric_name)
                        self._metric_fn_list[metric_name] = metric_object
                        self._metric_fn_kwargs[metric_name] = kwargs

                    except Exception:
                        raise Warning(
                            "{} not found in the evaluate library!".format(metric_name),
                            "Please check https://huggingface.co/evaluate-metric",
                        )

                if "aggregation" in metric_config:
                    agg_name = metric_config["aggregation"]
                    if type(agg_name) == str:
                        self._aggregation_list[metric_name] = AGGREGATION_REGISTRY[
                            agg_name
                        ]
                    elif callable(agg_name):
                        self._aggregation_list[metric_name] = metric_config[
                            "aggregation"
                        ]
                else:

                    INV_AGG_REGISTRY = {v: k for k, v in AGGREGATION_REGISTRY.items()}
                    metric_agg = DEFAULT_AGGREGATION_REGISTRY[metric_name]
                    eval_logger.warning(
                        f"metric {metric_name} is defined, but aggregation is not. "
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
                        f"metric {metric_name} is defined, but higher_is_better is not. "
                        f"using default "
                        f"higher_is_better={HIGHER_IS_BETTER_REGISTRY[metric_name]}"
                    )
                    self._higher_is_better[metric_name] = HIGHER_IS_BETTER_REGISTRY[
                        metric_name
                    ]

        self.download(self._config.dataset_kwargs)
        self._training_docs = None
        self._fewshot_docs = None

        if self._config.filter_list is not None:
            self._filters = []
            for filter_config in self._config.filter_list:
                for filter_pipeline in filter_config:
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
            self._filters = [build_filter_ensemble("none", [["take_first", None]])]

        if self._config.use_prompt is not None:
            eval_logger.info(f"loading prompt {self._config.use_prompt}")
            self.prompt = get_prompt(
                self._config.use_prompt, self.DATASET_PATH, self.DATASET_NAME
            )
        else:
            self.prompt = None

        if self.fewshot_docs() is not None:
            self.sampler = samplers.Sampler(
                list(self.fewshot_docs()), self, rnd=random.Random()
            )

    def download(self, dataset_kwargs=None):

        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            **dataset_kwargs if dataset_kwargs is not None else {},
        )

    def has_training_docs(self):
        if self._config.training_split is not None:
            return True
        else:
            return False

    def has_validation_docs(self):
        if self._config.validation_split is not None:
            return True
        else:
            return False

    def has_test_docs(self):
        if self._config.test_split is not None:
            return True
        else:
            return False

    def training_docs(self):
        if self._config.training_split is not None:
            return self.dataset[self._config.training_split]

    def validation_docs(self):
        if self._config.validation_split is not None:
            return self.dataset[self._config.validation_split]

    def test_docs(self):
        if self._config.test_split is not None:
            return self.dataset[self._config.test_split]

    def fewshot_docs(self):
        if self._config.fewshot_split is not None:
            return self.dataset[self._config.fewshot_split]
        else:
            if self._config.num_fewshot > 0:
                eval_logger.warning(
                    f"Task '{self._config.task}': "
                    "num_fewshot > 0 but fewshot_split is None. "
                    "using preconfigured rule."
                )
            return super().fewshot_docs()

    def should_decontaminate(self):
        return self._config.should_decontaminate

    def doc_to_decontamination_query(self, doc):
        if self._config.should_decontaminate:
            return utils.apply_template(self._config.doc_to_decontamination_query, doc)

    def _process_doc(self, doc):
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc

    def doc_to_text(self, doc):

        if self.prompt is not None:
            doc_to_text = self.prompt
        else:
            doc_to_text = self._config.doc_to_text

        if type(doc_to_text) == str:
            return utils.apply_template(doc_to_text, doc)
        elif callable(doc_to_text):
            return doc_to_text(doc)
        if hasattr(doc_to_text, "apply"):
            return doc_to_text.apply(doc)[0]
        else:
            print(type(doc_to_text))
            raise TypeError

    def doc_to_target(self, doc):

        if self.prompt is not None:
            doc_to_target = self.prompt
        else:
            doc_to_target = self._config.doc_to_target

        if type(doc_to_target) == str:
            return utils.apply_template(doc_to_target, doc)
        elif callable(doc_to_target):
            return doc_to_target(doc)
        elif hasattr(doc_to_target, "apply"):
            return doc_to_target.apply(doc)[1]
        else:
            raise TypeError

    def gold_alias(self, doc):
        # TODO: reevaluate if we need this. implemented to have a
        # processed version of answer to put into gsm8k exact_match scoring as ref.
        if self._config.gold_alias is not None:
            doc_to_target = self._config.gold_alias
        else:
            # doc_to_target = self._config.doc_to_target
            return self.doc_to_target(doc)

        if type(doc_to_target) == str:
            return utils.apply_template(doc_to_target, doc)
        elif callable(doc_to_target):
            return doc_to_target(doc)
        elif hasattr(doc_to_target, "apply"):
            return doc_to_target.apply(doc)[1]
        else:
            raise TypeError

    def construct_requests(self, doc, ctx, **kwargs):

        if self.OUTPUT_TYPE == "loglikelihood":
            arguments = (ctx, self.doc_to_target(doc))
        elif self.OUTPUT_TYPE == "loglikelihood_rolling":
            arguments = (self.doc_to_target(doc),)
        elif self.OUTPUT_TYPE == "multiple_choice":
            # we pass the user-defined answer_choices var (in aliases) and translate the result to a Python list.
            # TODO: any cleaner way to do this?
            choices = ast.literal_eval(
                utils.apply_template(
                    self._config.template_aliases + "{{answer_choices}}", doc
                )
            )
            request_list = [
                Instance(
                    request_type="loglikelihood",
                    doc=doc,
                    arguments=(ctx, " {}".format(choice)),
                    idx=i,
                    **kwargs,
                )
                for i, choice in enumerate(choices)
            ]
            # TODO: we should raise a warning telling users this will at most ~2x runtime.
            if "acc_mutual_info" in self._metric_fn_list.keys():
                # if we are calculating multiple choice accuracy
                # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

                # here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # in other words normalizing by subtracting the unconditional logprob of each choice.
                request_list.extend(
                    [
                        Instance(
                            request_type="loglikelihood",
                            doc=doc,
                            arguments=("", "{}".format(choice)),
                            idx=i,
                            **kwargs,
                        )
                        for i, choice in enumerate(choices)
                    ]
                )
            return request_list

        elif self.OUTPUT_TYPE == "greedy_until":
            arguments = (ctx, self._config.generation_kwargs)

        return Instance(
            request_type=self.OUTPUT_TYPE, doc=doc, arguments=arguments, idx=0, **kwargs
        )

    def process_results(self, doc, results):

        # if callable(self._config.process_results):
        #     return self._config.process_results(doc, results)

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

            lls, is_greedy = zip(*results)
            if self._config.gold_alias is not None:
                gold = int(self.gold_alias(doc))
            else:
                gold = int(self.doc_to_target(doc))

            pred = np.argmax(lls)
            # retrieve choices in List[str] form, to compute choice lengths, etc.
            choices = ast.literal_eval(
                utils.apply_template(
                    self._config.template_aliases + "{{answer_choices}}", doc
                )
            )
            if (
                2 * len(choices) == len(lls)
                and "acc_mutual_info" in self._metric_fn_list.keys()
            ):
                # then we are doing mutual info.
                # this stores the "dryrun" / unconditional answer loglikelihoods
                lls_unconditional = lls[1::2]
                assert len(lls_unconditional) == len(choices)
                # and this stores our "regular" conditional loglikelihoods
                lls = lls[::2]

            acc = 1.0 if np.argmax(lls) == gold else 0.0
            completion_len = np.array([float(len(i)) for i in choices])
            acc_norm = 1.0 if np.argmax(lls / completion_len) == gold else 0.0

            result_dict = {
                **({"acc": acc} if "acc" in use_metric else {}),
                **({"f1": (gold, pred)} if "f1" in use_metric else {}),
                **({"mcc": (gold, pred)} if "mcc" in use_metric else {}),
                **({"acc_norm": acc_norm} if "acc_norm" in use_metric else {}),
            }

            # TODO: set which normalization metrics should be reported, and calculate them
            if "exact_match" in self._metric_fn_list.keys():
                # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
                is_greedy = is_greedy[gold]  # take value for the gold answer
                result_dict["exact_match"] = int(is_greedy)

            if "acc_mutual_info" in use_metric:
                lls_mutual_info = [
                    ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional)
                ]
                acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
                result_dict["acc_mutual_info"] = acc_mutual_info

        elif self.OUTPUT_TYPE == "greedy_until":

            if self._config.gold_alias is not None:
                gold = self.gold_alias(doc)
            else:
                gold = self.doc_to_target(doc)

            for key, result in zip(self._metric_fn_list.keys(), results):
                _dict = self._metric_fn_list[key].compute(
                    references=[gold],
                    predictions=[result],
                    **self._metric_fn_kwargs[key],
                )

                result_dict = {**result_dict, **_dict}
        else:
            raise ValueError(
                f"Passed invalid output_type '{self.OUTPUT_TYPE}' ! Please use one of ",
                "'loglikelihood', 'loglikelihood_rolling', 'greedy_until', or 'multiple_choice'",
            )

        return result_dict

    def aggregation(self):
        return self._aggregation_list

    def higher_is_better(self):
        return self._higher_is_better


class MultipleChoiceTask(Task):

    OUTPUT_TYPE: str = "loglikelihood"

    def doc_to_target(self, doc):
        return " " + doc["choices"][doc["gold"]]

    def construct_requests(self, doc, ctx, **kwargs):
        # TODO: add mutual info here?
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                idx=i,
                **kwargs,
            )
            for i, choice in enumerate(doc["choices"])
        ]

    def process_results(self, doc, results):
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

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
        }


class PerplexityTask(Task):

    OUTPUT_TYPE = "loglikelihood_rolling"

    def has_training_docs(self):
        return False

    def fewshot_examples(self, k, rnd):
        assert k == 0
        return []

    def fewshot_context(self, doc, num_fewshot, rnd=None):
        assert (
            num_fewshot == 0
        ), "The number of fewshot examples must be 0 for perplexity tasks."
        assert (
            rnd is not None
        ), "A `random.Random` generator argument must be provided to `rnd`."

        return ""

    def higher_is_better(self):
        return {
            "word_perplexity": False,
            "byte_perplexity": False,
            "bits_per_byte": False,
        }

    def doc_to_decontamination_query(self, doc):
        return doc

    def doc_to_text(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc

    def construct_requests(self, doc, ctx, **kwargs):
        assert not ctx

        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=(self.doc_to_target(doc),),
            idx=0,
            **kwargs,
        )

    def process_results(self, doc, results):
        (loglikelihood,) = results
        words = self.count_words(self.doc_to_target(doc))
        bytes_ = self.count_bytes(self.doc_to_target(doc))
        return {
            "word_perplexity": (loglikelihood, words),
            "byte_perplexity": (loglikelihood, bytes_),
            "bits_per_byte": (loglikelihood, bytes_),
        }

    def aggregation(self):
        return {
            "word_perplexity": weighted_perplexity,
            "byte_perplexity": weighted_perplexity,
            "bits_per_byte": bits_per_byte,
        }

    @classmethod
    def count_bytes(cls, doc):
        return len(doc.encode("utf-8"))

    @classmethod
    def count_words(cls, doc):
        """Downstream tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))
