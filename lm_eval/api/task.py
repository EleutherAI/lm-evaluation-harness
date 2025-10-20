from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
)

import datasets
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.utils import Message, maybe_delimit, multiturn_to_singleturn
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.task import TaskConfig
from lm_eval.config.utils import process_field


ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "generate_until",
]

if TYPE_CHECKING:
    from collections.abc import Iterable

    import datasets

    DataSet = datasets.Dataset | Iterable[dict[str, Any]]
    DSplits = dict[str, DataSet]


eval_logger = logging.getLogger(__name__)


def format_turn(content: str, role: str):
    return {"role": role, "content": content}


class Task(ABC):
    VERSION: str = "Yaml"
    OUTPUT_TYPE: str = "generate_until"
    _registry: dict[str, type[Task]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register Task subclasses by their OUTPUT_TYPE."""
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "OUTPUT_TYPE") and cls.OUTPUT_TYPE:
            Task._registry[cls.OUTPUT_TYPE] = cls

    @classmethod
    def from_config(cls, config: TaskConfig | dict[str, Any]) -> Task:
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
            config if isinstance(config, TaskConfig) else TaskConfig(**config)
        )

        self.task = self.config.task
        self.OUTPUT_TYPE = self.config.output_type

        self.template = self.config.template
        self.sampler = self.config.fewshot_cfg.init_sampler(
            rnd=self.config.fewshot_cfg.rnd
        )
        self._filters = self.config.get_filters
        self._instances = None

        self.multiple_inputs = False
        self._dataset = None  # Lazy-loaded dataset

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
        doc: dict[str, str],
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: Callable[..., str] | None = None,
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
            for fs_doc in self.sampler.sample(
                n=num_fewshot,
                doc=doc
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
        self, doc: dict[str, Any], ctx: str | list[str] | dict[str, Any], **kwargs
    ) -> list[Instance] | Instance | None: ...

    @abstractmethod
    def process_results(self, doc: dict, results: list) -> dict[str, Any]: ...

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
            self._dataset = df(**(self.config.dataset_kwargs | self.config.metadata))
        else:
            assert self.config.dataset_path is not None, (
                "dataset_path must be set in TaskConfig"
            )
            self._dataset = datasets.load_dataset(
                path=self.config.dataset_path,
                name=self.config.dataset_name,
                **self.config.dataset_kwargs,
            )

    def fewshot_docs(self):
        docs = self.config.fewshot_cfg.get_docs(self.dataset)

        if docs is not None:
            return docs

        _num_fewshot = self.config.num_fewshot
        if isinstance(_num_fewshot, int) and _num_fewshot > 0:
            eval_logger.warning(
                f"[Task: {self.config.task}] "
                "num_fewshot > 0 but no fewshot source configured. "
                "Using preconfigured rule."
            )

            # Try splits in priority order
            for split_attr in ["training_split", "validation_split"]:
                if getattr(self.config, split_attr) is not None:
                    return self.get_docs(split_attr)

            # Fallback to test split
            eval_logger.warning(
                f"[Task: {self.config.task}] has_training_docs and has_validation_docs are False"
                ", using test_docs as fewshot_docs but this is not recommended."
            )
            return self.get_docs("test_split")

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
        if subset := getattr(self.config, subset):
            if self.config.process_docs is not None:
                assert self.dataset is not None, "dataset not set!"
                return self.config.process_docs(self.dataset[subset])
            return self.dataset[subset]

    def doc_to_text(
        self, doc: dict, *, doc_to_text: int | str | Callable[..., str] | None = None
    ) -> str | int:
        return process_field(doc, doc_to_text or self.config.doc_to_text)

    def doc_to_choice(
        self,
        doc: dict,
        *,
        doc_to_choice: str | list | dict | Callable[..., list[str]] | None = None,
    ) -> list[str]:
        return process_field(
            doc, doc_to_choice or self.config.doc_to_choice, lists=True
        )

    def doc_to_target(
        self, doc: dict, *, doc_to_target=None
    ) -> int | str | Sequence[int]:
        return process_field(
            doc,
            doc_to_target or self.config.doc_to_target,
            digits=self.config.doc_to_choice is not None,
        )

    def doc_to_audio(
        self, doc: dict, doc_to_audio: int | str | Callable[..., str] | None = None
    ):
        return process_field(
            doc, doc_to_audio or self.config.doc_to_audio, digits=False, lists=True
        )

    def doc_to_image(
        self, doc: dict, doc_to_image: int | str | Callable[..., str] | None = None
    ):
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
    def resolve_field(doc: dict[str, str], field: str | None = None):
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
    def sort_instances(instances: list[Instance]):
        """Sorts instances by doc_id and then by idx"""
        from collections import defaultdict

        instances_by_doc_id = defaultdict(list)
        for instance in instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        return instances_by_doc_id

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

    def training_docs(self) -> DataSet | None:
        return self.get_docs("training_split")


class GenerateTask(Task):
    OUTPUT_TYPE = "generate_until"

    def construct_requests(
        self, doc: dict[str, str], ctx: str | list[str] | list[dict[str, Any]], **kwargs
    ):
        assert self.OUTPUT_TYPE == "generate_until"
        arguments = (ctx, deepcopy(self.config.generation_kwargs))
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
            target=self.doc_to_target(doc),
            task_name=name,
            doc_id=doc_id,
            repeats=repeats,
            **instance_kwargs,
        )

    def process_results(self, doc: dict, results: list) -> dict[str, Any]:
        result_dict = {}
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

        return result_dict


class MultipleChoiceTask(Task):
    OUTPUT_TYPE = "loglikelihood"

    def construct_requests(
        self,
        doc: dict[str, str],
        ctx: str | list[str] | dict[str, Any],
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
        arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

        if "acc_mutual_info" in [m.metric_name for m in self.config._metric_list]:
            # if we are calculating multiple choice accuracy
            # using mutual information instead of raw loglikelihood as metric, need unconditional lls.

            # here mutual info refers to calculating
            # log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
            # in other words normalizing by subtracting the unconditional logprob of each choice.
            aux_arguments = [("", f"{target_delimiter}{choice}") for choice in choices]

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

    def process_results(self, doc: dict, results: list) -> dict[str, Any]:
        """
        Process results for multiple choice tasks.

        Args:
            doc: The document/example being evaluated
            results: List of (loglikelihood, is_greedy) tuples for each choice

        Returns:
            Dictionary of metric_name -> score
        """
        import numpy as np

        gold = self.doc_to_target(doc)

        # Handle mutual information metrics
        num_choices = len(self.doc_to_choice(doc))
        use_mutual_info = "acc_mutual_info" in [
            m.metric_name for m in self.config._metric_list
        ]

        if use_mutual_info:
            # Results are [conditional_lls..., unconditional_lls...]
            # Split them
            lls = [res[0] for res in results[:num_choices]]
            uncond_lls = [res[0] for res in results[num_choices:]]
            # Calculate mutual information: log(P(choice|ctx)) - log(P(choice))
            lls = [ll - uncond_ll for ll, uncond_ll in zip(lls, uncond_lls)]
        else:
            # Just extract loglikelihoods
            lls = [res[0] for res in results[:num_choices]]

        # Find the choice with highest loglikelihood
        pred = np.argmax(lls)

        # Compute metrics
        result_dict = {}
        for metric in self.config._metric_list:
            if metric.metric_name == "acc" or metric.metric_name == "acc_mutual_info":
                # Accuracy: did we pick the correct choice?
                result_dict[metric.metric_name] = int(pred == gold)
            elif metric.metric_name == "acc_norm":
                # Normalized accuracy (using length-normalized loglikelihoods)
                # Get the lengths from is_greedy flag (second element)
                # Actually, we need choice lengths - use the choices themselves
                choices = self.doc_to_choice(doc)
                # Length-normalize the loglikelihoods
                choice_lens = [len(choice) for choice in choices]
                lls_norm = [ll / max(length, 1) for ll, length in zip(lls, choice_lens)]
                pred_norm = np.argmax(lls_norm)
                result_dict[metric.metric_name] = int(pred_norm == gold)
            else:
                # For any custom metrics, try to compute them
                try:
                    result_score = metric.fn(
                        references=[gold],
                        predictions=[pred],
                        **metric.kwargs,
                    )
                    if isinstance(result_score, dict):
                        for k, v in result_score.items():
                            result_dict[k] = v
                    else:
                        result_dict[metric.metric_name] = result_score
                except (TypeError, KeyError):
                    # Fallback: try the old interface
                    try:
                        result_dict[metric.metric_name] = metric.fn([gold, pred])
                    except Exception:
                        eval_logger.warning(
                            f"Could not compute metric {metric.metric_name} for multiple choice"
                        )

        return result_dict


# Register "multiple_choice" as an alias for MultipleChoiceTask
# (both "loglikelihood" and "multiple_choice" use the same task class)
Task._registry["multiple_choice"] = MultipleChoiceTask


class PerplexityTask(Task):
    OUTPUT_TYPE = "loglikelihood_rolling"

    def construct_requests(
        self,
        doc: dict[str, str],
        ctx: str | list[str] | dict[str, Any],
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

    def process_results(self, doc: dict, results: list) -> dict[str, Any]:
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
