import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
)

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.utils import (
    Message,
    check_gold_index_error,
    format_turn,
    maybe_delimit,
    multiturn_to_singleturn,
)
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.task import TaskConfig
from lm_eval.config.utils import merge_dicts, process_field
from lm_eval.types import Results


if TYPE_CHECKING:
    from lm_eval.types import (
        ChatFormat,
        ChatTemplateProtocol,
        GenResult,
        MCResult,
        TaskDataSet,
    )

eval_logger = logging.getLogger(__name__)


class Task(ABC):
    """A task represents an entire benchmark, including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object that represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...}
    """

    VERSION: str = "Yaml"
    OUTPUT_TYPE: str | None = None
    DATASET_PATH: str | None = None
    DATASET_NAME: str | None = None
    CONFIG: dict[str, Any] = {}
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register Task subclasses by their OUTPUT_TYPE."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "OUTPUT_TYPE") and cls.OUTPUT_TYPE:
            Task._registry[cls.OUTPUT_TYPE] = cls

    @classmethod
    def from_config(cls, config: TaskConfig | dict[str, Any]):
        """
        Factory method to create the appropriate Task subclass based on output_type.

        Args:
            config: TaskConfig instance or dict with task configuration

        Returns:
            Instance of the appropriate Task subclass (GenerateTask, MultipleChoiceTask, etc.)
        """
        # Normalize to TaskConfig if needed
        if isinstance(config, dict):
            config = TaskConfig(**config)

        # Look up the appropriate Task class
        output_type = config.output_type
        if output_type not in cls._registry:
            raise ValueError(
                f"No Task class registered for output_type '{output_type}'. "
                f"Available types: {sorted(cls._registry.keys())}"
            )

        # Instantiate and return the appropriate subclass
        task_class = cls._registry[output_type]
        return task_class(config)

    def __init__(self, config: TaskConfig | dict[str, Any]):
        self.config: TaskConfig = (
            TaskConfig.from_arbitrary_dict(self.CONFIG)
            if self.CONFIG
            else (config if isinstance(config, TaskConfig) else TaskConfig(**config))
        )
        self.OUTPUT_TYPE = self.OUTPUT_TYPE or self.config.output_type
        assert self.OUTPUT_TYPE, "output_type must be set in TaskConfig or subclass"
        self.DATASET_NAME = self.DATASET_NAME or self.config.dataset_name
        self.DATASET_PATH = self.DATASET_PATH or self.config.dataset_path

        self.task = self.config.task

        self.template = self.config.template
        self.sampler = self.config._fewshot_cfg.init_sampler(
            rnd=self.config._fewshot_cfg.rnd
        )
        self._filters = self.config.get_filters()

        self.multiple_inputs = False
        # datasets are lazily loaded
        self._dataset = None
        self._fewshot_docs = None
        # created in build_all_requests
        self._instances = None

        ## outputs
        self._sample_scores: list = []
        self._aggregate_scores: dict[str, float] = defaultdict(float)

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
        """Build a set of Instances for a task and store them in task.instances"""

        # used with caching
        og_limit = limit

        cache_key = f"requests-{self.task}-{self.config.num_fewshot}shot-rank{rank}-world_size{world_size}"
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
        if self.config.repeats > 1 and self.OUTPUT_TYPE == "generate_until":
            flattened_instances = flattened_instances * self.config.repeats
        self._instances = flattened_instances

        if len(self._instances) == 0:
            raise ValueError("task.build_requests() did not find any docs!")

        if cache_requests and (not cached_instances or rewrite_requests_cache):
            save_to_cache(file_name=cache_key, obj=instances)

    def doc_iterator(
        self,
        *,
        rank: int = 0,
        limit: int | None = None,
        world_size: int = 1,
        samples: list[int] | None = None,
    ) -> Iterator[tuple[int, Any]]:
        assert (_eval_docs := self.eval_docs), "No eval_docs found!"
        if samples:
            assert self.eval_docs, "No eval_docs found!"
            n = len(list(_eval_docs))
            assert all(e < n for e in samples), (
                f"Elements of --samples should be in the interval [0,k-1] where k is the number of total examples. In this case, k={n}."
            )
            eval_logger.info(
                f"{self.config.task}: Evaluating on {len(samples)} examples"
            )
            _eval_docs = [x for i, x in enumerate(_eval_docs) if i in samples]
            limit = None

        limit = int(limit) if limit else None
        doc_iterator = utils.create_iterator(
            enumerate(_eval_docs),
            rank=int(rank),
            limit=limit,
            world_size=int(world_size),
        )
        return doc_iterator

    def fewshot_context(
        self,
        doc: dict[str, Any],
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: "ChatTemplateProtocol | None" = None,
        gen_prefix: str | None = None,
    ) -> str | list[str] | list[dict[str, Any]]:
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
        system_prompt = maybe_delimit(system_instruction, description, few_delim)
        # system_prompt = few_delim.join(filter(None, [system_instruction, description]))
        if system_prompt:
            messages.append(Message("system", system_prompt, tgt_delim))

        if num_fewshot > 0:
            for fs_doc in self.sampler.replace_df(self.fewshot_docs()).sample(
                n=num_fewshot,
                eval_doc=doc
                if self.config.fewshot_split == self.config.test_split
                else None,
            ):
                ## TODO: apply template in doc_to_qa
                q, c, a = self.apply_template_format(
                    self.doc_to_text(fs_doc),
                    self.doc_to_choice(fs_doc),
                    self.doc_to_target(fs_doc),
                )
                messages += self.doc_to_qa_message(
                    gen_prefix, q=q, c=c, a=a, tgt_delim=tgt_delim, few_delim=few_delim
                )

        q, c, a = self.apply_template_format(
            self.doc_to_text(doc),
            self.doc_to_choice(doc),
            self.doc_to_target(doc),
        )
        messages += self.doc_to_qa_message(
            gen_prefix,
            q=q,
            c=c,
            a=a,
            include_answer=False,
            tgt_delim=tgt_delim,
            few_delim=few_delim,
        )
        if apply_chat_template and chat_template:
            res = (
                [m.to_dict() for m in messages]
                if fewshot_as_multiturn
                else multiturn_to_singleturn(messages)
            )
        else:
            res = "".join(m.to_text() for m in messages)

        return res

    def doc_to_qa_message(
        self,
        gen_prefix: str | None = None,
        *,
        q: str | None = None,
        c: list[str] | None = None,
        a: str | int | None = None,
        include_answer: bool = True,
        tgt_delim=" ",
        few_delim="\n\n",
    ) -> list[Message]:
        """Return `[user, assistant?]` for a single doc."""
        assert isinstance(q, str), f"Context is not a string! : {q}"
        msgs = [Message("user", q, tgt_delim if include_answer or gen_prefix else "")]
        if include_answer:
            answer_text = c[a] if (c and isinstance(a, int)) else a
            assert isinstance(answer_text, str), f"Answer is not a string! : {a}"
            answer_text = maybe_delimit(gen_prefix, answer_text)
            msgs.append(Message("assistant", answer_text, few_delim))
        if gen_prefix:
            msgs.append(Message("assistant", gen_prefix))
        return msgs

    def apply_template_format(self, q: str, c: list[str] | None, a: str | int | None):
        if self.template is not None:
            return (
                self.template.format_prompt(q, c, a),
                self.template.format_choices(q, c, a),
                self.template.format_target(q, c, a),
            )
        else:
            return q, c, a

    @abstractmethod
    def construct_requests(
        self, doc: dict[str, Any], ctx: "ChatFormat | list[str]", **kwargs
    ) -> list[Instance] | Instance | None: ...

    @abstractmethod
    def process_results(
        self,
        doc: dict[str, Any],
        results: list,
        instances: list[Instance] | None = None,
        filter: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Process the results and return a dictionary where keys are the names of the metrics and values are the results of each metric.
        """
        if callable(process_res := self.config.process_results):
            return process_res(doc, results)
        else:
            return None

    @property
    def dataset(self):
        """Lazily load and return the dataset."""
        if self._dataset is None:
            self.download(self.config.dataset_kwargs)
        return self._dataset

    @property
    def eval_docs(self):
        if self.config.test_split:
            return self.get_docs("test_split")
        elif self.config.validation_split:
            return self.get_docs("validation_split")
        else:
            raise ValueError("Task dataset must have valid or test docs!")

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        import datasets
        from packaging.version import parse as vparse

        if dataset_kwargs and vparse(datasets.__version__) >= vparse("4.0.0"):
            dataset_kwargs.pop("trust_remote_code", None)

        _dataset_path = self.DATASET_PATH
        _dataset_name = self.DATASET_NAME

        self.config.dataset_kwargs, self.config.metadata = (
            self.config.dataset_kwargs or {},
            self.config.metadata or {},
        )

        if callable(df := self.config.custom_dataset):
            eval_logger.warning(
                f"{self.config.task}: Custom kwargs can be passed to `--metadata` in console (as json string) or to the TaskManager."
                + "\nFor example --metadata='{\"max_seq_lengths\":[4096, 8192]}'. For details see task Readme."
            )
            self._dataset = df(**(self.config.dataset_kwargs | self.config.metadata))
        else:
            assert _dataset_path is not None, (
                "dataset_path must be set in TaskConfig or class attribute"
            )
            df = datasets.load_dataset(
                path=_dataset_path,
                name=_dataset_name,
                **self.config.dataset_kwargs,
            )
        assert isinstance(df, dict)
        self._dataset = df

    def fewshot_docs(self):
        # Return cached fewshot docs if available
        if self._fewshot_docs:
            return self._fewshot_docs

        docs = self.config._fewshot_cfg.get_docs(self.dataset)

        if docs is not None:
            self._fewshot_docs = list(docs) if docs is not None else []
            return self._fewshot_docs

        _num_fewshot = self.config.num_fewshot
        if isinstance(_num_fewshot, int) and _num_fewshot > 0:
            eval_logger.warning(
                f"[Task: {self.config.task}] "
                "num_fewshot > 0 but no fewshot source configured. "
                "Using preconfigured rule."
            )

            # Try splits in priority order
            for split_attr in ["training_split", "validation_split"]:
                if (result := self.get_docs(split_attr)) is not None:
                    self._fewshot_docs = list(result)
                    return self._fewshot_docs

            # Fallback to test split
            eval_logger.warning(
                f"[Task: {self.config.task}] has_training_docs and has_validation_docs are False"
                ", using test_docs as fewshot_docs but this is not recommended."
            )
            if (result := self.get_docs("test_split")) is not None:
                self._fewshot_docs = list(result)
                return self._fewshot_docs

        self._fewshot_docs = []
        return self._fewshot_docs

    def apply_filters(self):
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters") and self.instances:
            for f in self._filters:
                f.ensemble.apply(self.instances)
            return self
        else:
            eval_logger.warning(
                "No filter defined or no instances, passing through instances"
            )
            return self

    def get_docs(self, subset: str):
        assert self.dataset is not None, "dataset not set!"
        if subset := getattr(self.config, subset):
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[subset])
            return self.dataset[subset]

    def doc_to_text(
        self,
        doc: dict[str, Any],
        *,
        doc_to_text: Callable[[dict[str, Any]], str] | str | None = None,
    ) -> str | int:
        return process_field(doc, doc_to_text or self.config.doc_to_text, default="")

    def doc_to_choice(
        self,
        doc: dict[str, Any],
        *,
        doc_to_choice: Callable[[dict[str, Any]], list[str]] | str | list | None = None,
    ) -> list[str]:
        return process_field(
            doc, doc_to_choice or self.config.doc_to_choice, lists=True, default=[]
        )

    def doc_to_target(
        self,
        doc: dict[str, Any],
        *,
        doc_to_target: Callable[[dict[str, Any]], str | int] | str | None = None,
    ) -> int | str | Sequence[int]:
        return process_field(
            doc,
            doc_to_target or self.config.doc_to_target,
            digits=self.config.doc_to_choice is not None,
            default="",
        )

    def doc_to_audio(
        self,
        doc: dict[str, Any],
        doc_to_audio: Callable[[dict[str, Any]], Any] | str | None = None,
    ) -> Any:
        return process_field(
            doc, doc_to_audio or self.config.doc_to_audio, digits=False, lists=True
        )

    def doc_to_image(
        self,
        doc: dict[str, Any],
        doc_to_image: Callable[[dict[str, Any]], Any] | str | None = None,
    ) -> Any:
        return process_field(
            doc, doc_to_image or self.config.doc_to_image, digits=False, lists=True
        )

    @property
    def instances(self) -> list[Instance] | None:
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances

    @staticmethod
    def resolve_field(doc: dict[str, Any], field: str | None = None):
        if field is not None:
            return doc[field] if field in doc else utils.apply_template(field, doc)

    def get_config(self, key: str) -> Any:
        """Get a configuration value by key."""
        return getattr(self.config, key, None)

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
        return self.config.to_dict()

    def set_config(self, key: str, value: Any, update: bool = False) -> None:
        """Set or update the configuration for a given key."""
        if update:
            current_value = getattr(self.config, key, {})
            if not isinstance(current_value, dict):
                raise TypeError(
                    f"Expected a dict for key '{key}', got {type(current_value).__name__} instead."
                )
            current_value.update(value)
        else:
            setattr(self.config, key, value)

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        """Set the random seed for fewshot sampling."""
        if hasattr(self, "sampler"):
            self.sampler.set_rnd(seed)

    def override_metric(self, metric_name: str) -> None:
        """
        Override the default metrics used for evaluation with custom metrics.

        Parameters:
        - metric_name (str): The name of the custom metric to override. Should be registered in api.metrics.
        """
        from lm_eval.config.metric import MetricConfig

        self.config.metric_list = [MetricConfig(name=metric_name)]
        self.config.process_results = lambda *args: {"bypass": 0}

    @staticmethod
    def sort_instances(instances: list[Instance]) -> dict[str, list[Instance]]:
        """Sorts instances by doc_id and then by idx"""
        from collections import defaultdict

        instances_by_doc_id = defaultdict(list)
        for instance in instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        return instances_by_doc_id

    def process_instances(self, instances: list[Instance] | None = None):
        """Primary method for processing instances to compute metrics.

        This is the main entry point for result processing. It iterates over
        all filters and documents, delegating to _process_doc_instances for
        type-specific processing.

        Args:
            instances: List of instances to process. If None, uses self._instances

        Returns:
            Dictionary mapping (metric_name, filter_name) -> list of scores
        """
        _instances = self.sort_instances(instances or self._instances)
        if not _instances:
            return {}

        results = [Results.create(inst) for inst in _instances.values()]
        valid_metrics = [
            (filter, metric)
            for filter in self._filters
            for metric in filter.metric_list
            if metric is not None and metric.fn is not None
        ]
        for _res in results:
            for filter, metric in valid_metrics:
                metric_result = metric.fn(**_res.to_metric_inputs())
                _res.scores[(metric.name, filter.name)] = metric_result

        # for filter in self._filters:
        #     for _metric in filter.metric_list:
        #         if _metric is not None and _metric.fn is not None:
        #             metric_result = _metric.fn(result_obj.to_metric_inputs())
        #             result_obj.scores[(_metric.name, filter.name)] = metric_result
        #
        # results = []
        # for filter in self._filters:
        #     for _metric in filter.metric_list:
        #         if _metric is not None and _metric.fn is not None:
        #             for inst in _instances.values():
        #                 result_obj = Results.create(inst)
        #                 metric_result = _metric.fn(result_obj.to_metric_inputs())
        #                 result_obj.scores[(_metric.name, filter.name)] = metric_result
        #                 results.append(result_obj)
        self._sample_scores = results
        return results

    @abstractmethod
    def _process_doc_instances(
        self, instances: list[Instance], filter_name: str
    ) -> dict[str, Any]:
        """Process instances for a single document.

        This method must be overridden in subclasses to provide type-specific
        processing logic. It receives all instances for a single doc_id and
        should return a dictionary of metric scores.

        Args:
            instances: List of Instance objects for the same doc_id
            filter_name: Name of the filter to use for filtered responses

        Returns:
            Dictionary mapping metric_name -> score
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _process_doc_instances"
        )

    def compute_metrics(self, instances: list[Instance] | None = None) -> dict:
        """Deprecated alias for process_instances.

        This method is kept for backward compatibility. New code should use
        process_instances() instead.

        Args:
            instances: List of instances to process

        Returns:
            Dictionary mapping (metric_name, filter_name) -> list of scores
        """
        return self.process_instances(instances)

    def aggregation(self) -> dict:
        """Return aggregation functions for each metric."""
        return {k.name: k.aggregation_fn for k in self.config._metric_list}

    def higher_is_better(self) -> dict:
        """Return whether higher is better for each metric."""
        return {k.name: k.higher_is_better for k in self.config._metric_list}

    @property
    def task_name(self) -> str | None:
        """Return the task name."""
        return getattr(self.config, "task", None)

    def training_docs(self) -> "TaskDataSet | None":
        return self.get_docs("training_split")


class GenerateTask(Task):
    OUTPUT_TYPE = "generate_until"

    def construct_requests(
        self, doc: dict[str, Any], ctx: str | list[str] | list[dict[str, Any]], **kwargs
    ):
        assert self.OUTPUT_TYPE == "generate_until"
        arguments = (ctx, deepcopy(self.config.generation_kwargs))
        name, doc_id, repeats = kwargs.get("metadata", ("", 0, 1))

        # Filter out chat_template and metadata from kwargs
        instance_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("chat_template", "apply_chat_template")
        }

        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=arguments,
            idx=0,
            target=self.doc_to_target(doc),
            task_name=name,
            doc_id=doc_id,
            repeats=repeats,
            **instance_kwargs,
        )

    def _process_doc_instances(
        self, instances: list[Instance], filter_name: str
    ) -> dict[str, Any]:
        """Process generation instances for a single doc."""
        # Check for custom process_results
        if callable(self.config.process_results):
            # Extract data for backward compatibility
            doc = instances[0].doc
            results = [inst.filtered_resps[filter_name] for inst in instances]
            return self.config.process_results(doc, results)

        # Standard path using GenResult
        from lm_eval.types import GenResult

        gen_result = GenResult.from_instances(instances, filter_name)
        gen_result.repeats = self.config.repeat_cfg.repeats

        return self._compute_generation_metrics(gen_result)

    def _compute_generation_metrics(self, gen_result: "GenResult") -> dict[str, Any]:
        """Compute metrics from GenResult with repeat reduction."""
        result_dict = {}
        gold = gen_result.target

        # Step 1: Compute metrics for each generation
        per_generation_scores = defaultdict(list)

        for generation in gen_result.results:
            for metric in self.config._metric_list:
                if metric.fn is not None:
                    try:
                        score = metric.fn(
                            references=[gold] if not isinstance(gold, list) else gold,
                            predictions=[generation],
                            **metric.kwargs,
                        )
                    except TypeError:
                        # Handle metrics with different interfaces
                        score = metric.fn([gold, generation])

                    if isinstance(score, dict):
                        # Multiple metrics from same function
                        for k, v in score.items():
                            per_generation_scores[k].append(v)
                    else:
                        per_generation_scores[metric.name].append(score)

        # Step 2: Handle repeat reduction
        if gen_result.is_repeated and self.config.repeat_cfg.repeats > 1:
            reduced_scores = {}
            for metric_name, scores in per_generation_scores.items():
                # Add raw scores
                result_dict[metric_name] = scores

                # Apply reduction if we have the right number of scores
                if len(scores) == self.config.repeat_cfg.repeats:
                    reduced_value = self.config.repeat_cfg.reducer(scores)

                    # Add reduced metric with special name
                    reduced_name = f"{metric_name}_{self.config.repeat_cfg.metric_name}"
                    result_dict[reduced_name] = reduced_value
                else:
                    eval_logger.warning(
                        f"Repeat metric {metric_name} has {len(scores)} scores, "
                        f"expected {self.config.repeat_cfg.repeats}. Skipping reduction."
                    )
        else:
            # No repeats - flatten single scores
            for k, v in per_generation_scores.items():
                result_dict[k] = v[0] if len(v) == 1 else v

        return result_dict

    def process_results(
        self,
        doc: dict[str, Any],
        results: list[str],
        instances: list[Instance] | None = None,
    ) -> dict[str, Any]:
        """Legacy compatibility method for processing results.

        This method is kept for backward compatibility. When instances are
        available, it delegates to the new _process_doc_instances method.
        """
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)

        # Use new flow if instances are provided
        if instances:
            return self._process_doc_instances(instances, "default")

        # Fallback to legacy implementation
        # This shouldn't normally be reached in the new flow
        result_dict = {}
        gold = self.doc_to_target(doc)
        _res: list[dict] = []
        for result in results:
            for metric in self.config._metric_list:
                if metric.fn is not None:
                    try:
                        result_score = metric.fn(
                            references=[gold] if not isinstance(gold, list) else gold,
                            predictions=[result],
                            **metric.kwargs,
                        )
                    except TypeError:
                        result_score = metric.fn([gold, result])
                    if isinstance(result_score, dict):
                        _res.append(result_score)
                    else:
                        _res.append({metric.name: result_score})
        result_dict = merge_dicts(_res)
        if self.config.repeat_cfg.repeats > 1:
            for k, v in result_dict.items():
                _res = {}
                if len(v) != self.config.repeat_cfg.repeats:
                    eval_logger.warning(
                        f"Repeat metric {k} is not compatible with {self.config.repeat_cfg.repeats} repeats. Passing through to aggregation, with repeat reduction"
                    )
                else:
                    _res[f"{k}_{self.config.repeat_cfg.metric_name}"] = (
                        self.config.repeat_cfg.reducer(v)
                    )

        return result_dict | _res


class MultipleChoiceTask(Task):
    OUTPUT_TYPE = "loglikelihood"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        apply_chat_template: bool = False,
        **kwargs,
    ):
        choices = self.doc_to_choice(doc)
        target_delimiter = (
            ""
            if (apply_chat_template and not self.config.gen_prefix)
            else self.config.target_delimiter
        )
        # if self.multiple_inputs:
        #     # If there are multiple inputs, assume only one choice
        #     arguments = [(_ctx, f"{target_delimiter}{choices[0]}") for _ctx in ctx]
        # else:
        # Otherwise they are placed in the continuation
        if isinstance(ctx, list) and isinstance(ctx[0], str):
            if isinstance(ctx[0], str):
                # If there are multiple inputs, assume only one choice
                arguments = [(_ctx, f"{target_delimiter}{choices[0]}") for _ctx in ctx]
            elif isinstance(ctx[0], dict):
                # If there are multiple inputs, assume only one choice
                arguments = [
                    (ctx + [format_turn(cont, "assistant")]) for cont in choices
                ]
            if "acc_mutual_info" in [m.metric_name for m in self.config._metric_list]:
                raise ValueError(
                    "acc_mutual_info not supported for multiple inputs, or for custom tokenizers."
                )
        else:
            arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

            if "acc_mutual_info" in [m.metric_name for m in self.config._metric_list]:
                # if we are calculating multiple choice accuracy
                # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

                # here mutual info refers to calculating
                # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
                # in other words normalizing by subtracting the unconditional logprob of each choice.
                aux_arguments = [
                    (self.config.unconditional_context, f"{target_delimiter}{choice}")
                    for choice in choices
                ]

                arguments.extend(aux_arguments)

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

        # Filter out chat_template from kwargs (it's not accepted by Instance)
        instance_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("chat_template", "metadata")
        }

        # Create instances for all loglikelihood requests (both regular and mutual info)
        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                idx=i,
                target=self.doc_to_target(doc),
                task_name=kwargs.get("metadata", (None, None, None))[0],
                doc_id=kwargs.get("metadata", (None, None, None))[1],
                repeats=kwargs.get("metadata", (None, None, None))[2],
                **instance_kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

        return request_list

    def _process_doc_instances(
        self, instances: list[Instance], filter_name: str
    ) -> dict[str, Any]:
        """Process multiple choice instances for a single doc."""
        # Check for custom process_results
        if callable(self.config.process_results):
            # Extract data for backward compatibility
            doc = instances[0].doc
            results = [inst.filtered_resps[filter_name] for inst in instances]
            return self.config.process_results(doc, results)

        # Standard path using MCResult
        from lm_eval.types import MCResult

        use_metric = [m.metric_name for m in self.config._metric_list]
        acc_mutual_info = "acc_mutual_info" in use_metric

        mc_result = MCResult.from_instances(instances, acc_mutual_info=acc_mutual_info)
        return self._compute_mc_metrics(mc_result)

    def _compute_mc_metrics(self, mc_result: "MCResult", metric_name) -> dict[str, Any]:
        """Compute metrics from MCResult."""
        import numpy as np

        # Get list of metric names we need to compute
        use_metric = [m.metric_name for m in self.config._metric_list]

        # Compute predictions
        pred = np.argmax(mc_result.lls)
        pred_norm = np.argmax(mc_result.lls / mc_result.char_lens)

        gold = mc_result.target

        # Compute accuracy metrics
        acc = 1.0 if pred == gold else 0.0
        acc_norm = 1.0 if pred_norm == gold else 0.0
        exact_match = (
            int(mc_result.is_greedy[gold])
            if (isinstance(gold, int) and gold != -100)
            else 0
        )

        # Compute normalized probabilities for brier score
        prob_norm = utils.softmax(mc_result.lls)

        # Build result dictionary with only the metrics we need
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

        # Handle mutual information accuracy
        if "acc_mutual_info" in use_metric:
            if not mc_result.lls_mutual_info:
                raise ValueError(
                    "acc_mutual_info requires unconditional loglikelihoods, "
                    "but they were not computed. This is a configuration error."
                )
            acc_mutual_info = (
                1.0 if np.argmax(mc_result.lls_mutual_info) == gold else 0.0
            )
            result_dict["acc_mutual_info"] = acc_mutual_info

        return result_dict

    def process_results(
        self,
        doc: dict[str, Any],
        results: list,
        instances: list[Instance] | None = None,
        filter: str | None = None,
    ) -> dict[str, Any]:
        """Legacy compatibility method for processing results.

        This method is kept for backward compatibility. When instances are
        available, it delegates to the new _process_doc_instances method.

        Args:
            doc: The document/example being evaluated
            results: List of (loglikelihood, is_greedy) tuples for each choice
            instances: List of instances for this doc (new flow)
            filter: Filter name to use

        Returns:
            Dictionary of metric_name -> score
        """
        # Check for custom process_results first
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)

        # Use new flow if instances are provided
        if instances:
            return self._process_doc_instances(
                instances, filter if filter else "default"
            )

        # Fallback to legacy implementation
        # This shouldn't normally be reached in the new flow

        # Extract loglikelihoods and is_greedy flags
        lls, is_greedy = zip(*results)

        # Retrieve choices in List[str] form, to compute choice lengths, etc.
        choices = self.doc_to_choice(doc)
        completion_len = np.array([float(len(i)) for i in choices])

        # Get list of metric names we need to compute
        use_metric = [m.metric_name for m in self.config._metric_list]

        # Handle mutual information if needed
        lls_unconditional = None
        if 2 * len(choices) == len(lls) and "acc_mutual_info" in use_metric:
            lls_unconditional = lls[len(choices) :]
            if len(lls_unconditional) != len(choices):
                raise ValueError
            lls = lls[: len(choices)]

        # Compute predictions
        pred = np.argmax(lls)
        pred_norm = np.argmax(lls / completion_len)

        # Get gold label and validate
        gold = self.doc_to_target(doc)
        if isinstance(gold, Sequence) and not isinstance(gold, (str, list)):
            gold = list(gold)
        gold, gold_index_error = check_gold_index_error(choices, gold)

        if gold_index_error:
            eval_logger.warning(
                f"Label index was not in within range of available choices,"
                f"Sample:\n\n{doc}\n\n"
            )

        # Compute accuracy metrics
        acc = 1.0 if pred == gold else 0.0
        acc_norm = 1.0 if pred_norm == gold else 0.0
        exact_match = (
            int(is_greedy[gold]) if (isinstance(gold, int) and gold != -100) else 0
        )

        # Compute normalized probabilities for brier score
        prob_norm = utils.softmax(lls)

        # Build result dictionary with only the metrics we need
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

        # Handle mutual information accuracy
        if "acc_mutual_info" in use_metric:
            if lls_unconditional is None:
                raise ValueError(
                    "acc_mutual_info requires unconditional loglikelihoods, "
                    "but they were not computed. This is a configuration error."
                )
            lls_mutual_info = [
                ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional)
            ]
            acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
            result_dict["acc_mutual_info"] = acc_mutual_info

        return result_dict


# Register "multiple_choice" as an alias for MultipleChoiceTask
# (both "loglikelihood" and "multiple_choice" use the same task class)
Task._registry["multiple_choice"] = MultipleChoiceTask


class PerplexityTask(Task):
    OUTPUT_TYPE = "loglikelihood_rolling"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        **kwargs,
    ) -> Instance:
        """
        Construct a loglikelihood_rolling request for perplexity evaluation.

        For rolling loglikelihood, we compute the log probability of the entire
        target sequence using a sliding window approach.
        """
        # For loglikelihood_rolling, we pass the target text
        target = self.doc_to_target(doc)

        # The arguments are just the target text (no context needed for rolling LL)
        arguments = (target,)

        name, doc_id, repeats = kwargs.get("metadata", ("", 0, 1))

        # Filter out chat_template and metadata from kwargs
        instance_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("chat_template", "metadata", "apply_chat_template")
        }

        return Instance(
            request_type=self.OUTPUT_TYPE,
            doc=doc,
            arguments=arguments,
            idx=0,
            target=target,
            task_name=name,
            doc_id=doc_id,
            repeats=repeats,
            **instance_kwargs,
        )

    def _process_doc_instances(
        self, instances: list[Instance], filter_name: str
    ) -> dict[str, Any]:
        """Process perplexity instances for a single doc."""
        # Check for custom process_results
        if callable(self.config.process_results):
            # Extract data for backward compatibility
            doc = instances[0].doc
            results = [inst.filtered_resps[filter_name] for inst in instances]
            return self.config.process_results(doc, results)

        # Standard perplexity processing
        # Extract the first result (should only be one for rolling LL)
        result = instances[0].filtered_resps.get(filter_name, instances[0].resps)
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        loglikelihood = result[0] if isinstance(result, tuple) else result

        target = instances[0].target
        assert isinstance(target, str), (
            "Require target to be a string for loglikelihood_rolling"
        )

        use_metric = [m.metric_name for m in self.config._metric_list]
        _words = self.count_words(target)
        _bytes = self.count_bytes(target)

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

    def process_results(
        self,
        doc: dict[str, Any],
        results: list,
        instances: list[Instance] | None = None,
    ) -> dict[str, Any]:
        """Legacy compatibility method for processing results.

        This method is kept for backward compatibility. When instances are
        available, it delegates to the new _process_doc_instances method.
        """
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)

        # Use new flow if instances are provided
        if instances:
            return self._process_doc_instances(instances, "default")

        # Fallback to legacy implementation
        (loglikelihood, *_) = results
        assert isinstance(_target := self.doc_to_target(doc), str), (
            "Require target to be a string for loglikelihood_rolling"
        )
        use_metric = list(m.metric_name for m in self.config._metric_list)
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

    @staticmethod
    def count_bytes(doc: str) -> int:
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @staticmethod
    def count_words(doc: str) -> int:
        """Downstream loglikelihood_rolling perplexity tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))
