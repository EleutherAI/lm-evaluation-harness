# ruff: noqa: F821, F841
from __future__ import annotations

import abc
import ast
import logging
import random
import re
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import numpy as np
from distlib.util import cached_property
from tqdm import tqdm
from typing_extensions import TypedDict

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.instance import AdditionalArgs, Instance
from lm_eval.api.utils import (
    Message,
    ends_with_whitespace,
    maybe_delimit,
    multiturn_to_singleturn,
    normalize_to_list,
    random_task_id,
    requires_delimiter,
)
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.task import TaskConfig
from lm_eval.config.utils import process_field


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import datasets

    from lm_eval.config.task import FewshotConfig
    from lm_eval.types import ChatTemplate, OutputType

eval_logger = logging.getLogger(__name__)


class METADATA(TypedDict, total=True, closed=False, extra_items=Any):
    task: str
    doc_id: int
    repeats: int


class Task:
    """A task represents an entire benchmark, including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object that represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...}
    """

    VERSION: str = "Yaml"
    OUTPUT_TYPE: OutputType | None = None
    DATASET_PATH: str | None = None
    DATASET_NAME: str | None = None
    MULTIMODAL: bool = False
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

    @staticmethod
    def count_bytes(doc):
        """Used for byte-level perplexity metrics in rolling loglikelihood"""
        return len(doc.encode("utf-8"))

    @staticmethod
    def count_words(doc):
        """Downstream loglikelihood_rolling perplexity tasks with custom word boundaries should override this!"""
        return len(re.split(r"\s+", doc))

    def __init__(self, config: TaskConfig | dict[str, Any]):
        self._config: TaskConfig = (
            TaskConfig(**self.CONFIG)
            if self.CONFIG
            else (config if isinstance(config, TaskConfig) else TaskConfig(**config))
        )
        # TODO: make this str rather than str|None
        self.task = self._config.task
        assert self.task is not None
        self.OUTPUT_TYPE = self.OUTPUT_TYPE or self._config.output_type
        assert self.OUTPUT_TYPE, "output_type must be set in TaskConfig or subclass"
        self._dataset_name = self.DATASET_NAME or self._config.dataset_name
        self._dataset_path = self.DATASET_PATH or self._config.dataset_path
        self._fewshot_cfg: FewshotConfig = self._config.fewshot_config

        self._multiple_inputs = self._config.multiple_inputs
        self._multimodal = self.MULTIMODAL or bool(
            self.config.doc_to_audio or self.config.doc_to_image
        )

        # TODO: make lazy
        if (_fs_docs := self.fewshot_docs()) is not None:
            config_sampler: str | type[samplers.ContextSampler] = (
                self._fewshot_cfg.sampler if self._fewshot_cfg else "default"
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
            self.sampler: samplers.ContextSampler = sampler_cls(fewshot_docs, rnd=None)

        # lazy load dataset
        self._dataset = None
        self._instances = None

    ### Dataset Loading and Doc Access ###

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        import datasets
        from packaging.version import parse as vparse

        if dataset_kwargs and vparse(datasets.__version__) >= vparse("4.0.0"):
            dataset_kwargs.pop("trust_remote_code", None)

        self._config.dataset_kwargs, self._config.metadata = (
            self._config.dataset_kwargs or {},
            self._config.metadata or {},
        )

        if callable(df := self._config.custom_dataset):
            eval_logger.warning(
                f"{self._config.task}: Custom kwargs can be passed to `--metadata` in console (as json string) or to the TaskManager."
                + "\nFor example --metadata='{\"max_seq_lengths\":[4096, 8192]}'. For details see task Readme."
            )
            self._dataset = df(**(self._config.dataset_kwargs | self._config.metadata))
        else:
            assert self._dataset_path is not None, (
                "dataset_path must be set in TaskConfig or class attribute"
            )
            df = datasets.load_dataset(
                path=self._dataset_path,
                name=self._dataset_name,
                **self._config.dataset_kwargs,
            )
        assert isinstance(df, dict)
        self._dataset = df

    @property
    def dataset(self):
        """Lazily load and return the dataset."""
        if self._dataset is None:
            self.download(self.config.dataset_kwargs)
        return self._dataset

    @property
    def config(self) -> TaskConfig:
        """Returns the TaskConfig associated with this class."""
        return self._config

    # def has_training_docs(self) -> bool:
    #     return self.config.training_split is not None
    #
    # def has_validation_docs(self) -> bool:
    #     return self.config.validation_split is not None
    #
    # def has_test_docs(self) -> bool:
    #     return self.config.test_split is not None

    def training_docs(self) -> datasets.Dataset | None:
        if self.config.training_split is not None:
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.training_split]
                )
            return self.dataset[self.config.training_split]

    def validation_docs(self) -> datasets.Dataset | None:
        if self.config.training_split is not None:
            if self.config.process_docs is not None:
                return self.config.process_docs(
                    self.dataset[self.config.validation_split]
                )
            return self.dataset[self.config.validation_split]

    def test_docs(self) -> datasets.Dataset | None:
        if self.config.test_split is not None:
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[self.config.test_split])
            return self.dataset[self.config.test_split]

    def fewshot_docs(self):
        if (_df := self._fewshot_cfg.get_docs(self.dataset)) is not None:
            self._fewshot_docs = list(_df)
            return _df
        else:
            if (_shots := self._config.num_fewshot) is not None and (_shots > 0):
                eval_logger.warning(
                    f"[Task: {self._config.task}] "
                    f"num_fewshot > 0 but fewshot_split is None. "
                    "using preconfigured rule."
                )
                # Try splits in priority order
                _df = self.training_docs() or self.validation_docs()
                if _df is not None:
                    self._fewshot_docs = list(_df)
                    return self._fewshot_docs

                # Fallback to test split
                eval_logger.warning(
                    f"[Task: {self._config.task}] has_training_docs and has_validation_docs are False"
                    ", using test_docs as fewshot_docs but this is not recommended."
                )
                if (_df := self.test_docs()) is not None:
                    self._fewshot_docs = list(_df)
                    return self._fewshot_docs

                self._fewshot_docs = []
                return self._fewshot_docs

    @property
    def eval_docs(self) -> datasets.Dataset | list[dict[str, Any]]:
        _df = self.test_docs() or self.validation_docs()
        if _df is None:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have valid or test docs!"
            )
        return _df

    def get_docs(self, subset: str) -> list[dict[str, Any]] | None:
        assert self.dataset is not None, "dataset not set!"
        if subset := getattr(self.config, subset):
            if self.config.process_docs is not None:
                return self.config.process_docs(self.dataset[subset])
            return self.dataset[subset]

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

    ### Context Construction ###

    def fewshot_context(
        self,
        doc: dict,
        num_fewshot: int,
        system_instruction: str | None = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        chat_template: ChatTemplate | None = None,
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
                if self._fewshot_cfg.split == self.config.test_split
                else None,
            ):
                q, c, a = (
                    self.doc_to_text(fs_doc, self._fewshot_cfg.doc_to_text),
                    self.doc_to_choice(fs_doc, self._fewshot_cfg.doc_to_choice)
                    if self._fewshot_cfg.doc_to_choice
                    else None,
                    self.doc_to_target(fs_doc, self._fewshot_cfg.doc_to_target),
                )
                _gen_prefix = self.resolve_field(doc, self._fewshot_cfg.gen_prefix)
                # for multiple inputs, q: int, c: list[str], target: str
                # TODO: fix this hacky way of handling multiple inputs
                if self._multiple_inputs:
                    q = cast("str", c[q])  # type: ignore
                    c = None
                # TODO: fix types
                messages += self.build_qa_turn(
                    q=q,
                    c=c,
                    a=a,
                    gen_prefix=_gen_prefix,
                    tgt_delim=self._fewshot_cfg.target_delimiter,
                    few_delim=self._fewshot_cfg.fewshot_delimiter,
                )

        q, c, a = (
            self.doc_to_text(doc),
            self.doc_to_choice(doc) if self.config.doc_to_choice else None,
            self.doc_to_target(doc),
        )
        if self._multiple_inputs:
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

    @abc.abstractmethod
    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance] | Instance: ...

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
        chat_template: ChatTemplate | None = None,
        tokenizer_name: str = "",
    ) -> list[Instance]:
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
            return flattened_instances

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
                metadata={
                    "task": self.task_name,
                    "doc_id": doc_id,
                    "repeats": self.config.repeats,
                },
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

        return flattened_instances

    @property
    def instances(self) -> list[Instance]:
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances or []

    ### Doc to Text/Target/Choice/Multimodal Parsing Methods ##

    def doc_to_text(self, doc, doc_to_text=None) -> str | list[str]:
        doc_to_text = (
            doc_to_text if doc_to_text is not None else self.config.doc_to_text
        )
        y = process_field(doc, doc_to_text, digits=True, lists=False, default=None)
        return y

    def doc_to_choice(self, doc, doc_to_choice=None) -> list[str]:
        doc_to_choice = (
            doc_to_choice if doc_to_choice is not None else self.config.doc_to_choice
        )
        y = process_field(doc, doc_to_choice, digits=True, lists=True, default=[])
        return y

    def doc_to_target(self, doc, doc_to_target=None) -> str | int | list[str] | None:
        doc_to_target = (
            doc_to_target if doc_to_target is not None else self.config.doc_to_target
        )
        y = process_field(doc, doc_to_target, digits=True, lists=True, default=None)
        return y

    def doc_to_prefix(self, doc):
        if (gen_prefix := self.config.gen_prefix) is not None:
            if gen_prefix in self.features:
                return doc[gen_prefix]
            else:
                return utils.apply_template(gen_prefix, doc)
        return None

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

    @cached_property
    def is_multomodal(self):
        return (
            self.config.doc_to_image is not None or self.config.doc_to_audio is not None
        )

    def _build_multimodal_kwargs(self, doc, **kwargs) -> AdditionalArgs | None:
        assert self.MULTIMODAL, (
            "This method should only be called for multimodal tasks, but this task is not multimodal according to its config."
        )
        multimodal_kwargs: AdditionalArgs = {}
        if self.config.doc_to_image and (images := self.doc_to_image(doc)) is not None:
            multimodal_kwargs["visual"] = images
        if self.config.doc_to_audio and (audio := self.doc_to_audio(doc)) is not None:
            multimodal_kwargs["audio"] = audio
        return multimodal_kwargs or None

    ### Result Instance Processing ###
    def apply_filters(self) -> list[Instance] | None:
        """Iterates over FilterEnsembles and applies them to instances"""
        if hasattr(self, "_filters"):
            for f in self._filters:
                f.apply(self._instances)
        else:
            eval_logger.warning("No filter defined, passing through instances")
            return self._instances

    def process_instances(self, instances: list[Instance]):
        _instances = self._sort_instances(instances)

    def _process_results(
        self,
        doc: dict[str, Any],
        results: list[Any],
    ) -> dict[str, list[Any]] | None:
        """
        Process the results and return a dictionary where keys are the names of the metrics and values are the results of each metric.
        """
        if callable(process_res := self.config.process_results):
            return normalize_to_list(process_res(doc, results))
        else:
            return None

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

            if self.multiple_inputs:
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

    @staticmethod
    def _sort_instances(
        instances: list[Instance] | None = None,
    ) -> dict[int, list[Instance]]:
        """Sorts instances by doc_id and then by idx"""
        if not instances:
            return {}
        from collections import defaultdict

        instances_by_doc_id: dict[int, list[Instance]] = defaultdict(list)
        for instance in instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        return instances_by_doc_id

    ## Some public methods and utilities ##
    @property
    def task_name(self) -> str:
        return getattr(self.config, "task", random_task_id())

    @cached_property
    def id(self):
        from random import Random

        return Random(f"{self.task_name}").randint(0, 2**32)

    def aggregation(self) -> dict:
        return self._aggregation_list

    def higher_is_better(self) -> dict:
        return self._higher_is_better

    @staticmethod
    def _overload_process_results(
        doc, results, fn: Callable[[dict[str, Any], Any], dict] | None
    ):
        return fn(doc, results) if fn is not None else results

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

    @staticmethod
    def resolve_field(doc: dict[str, Any], field: str | None = None):
        if field:
            return doc[field] if field in doc else utils.apply_template(field, doc)

    # def override_metric(self, metric_name: str) -> None:
    #     """
    #     Override the default metrics used for evaluation with custom metrics.
    #
    #     Parameters:
    #     - metric_name (str): The name of the custom metric to override. Should be registered in api.metrics.
    #     """
    #     (
    #         self._metric_fn_list,
    #         self._aggregation_list,
    #         self._metric_fn_kwargs,
    #         self._higher_is_better,
    #     ) = ({}, {}, {}, {})
    #     self._metric_fn_list[metric_name] = get_metric(metric_name)
    #     self._aggregation_list[metric_name] = get_metric_aggregation(metric_name)
    #     self._higher_is_better[metric_name] = is_higher_better(metric_name)
    #     self._metric_fn_kwargs[metric_name] = {}
    #     if not isinstance(self, Task):
    #         self.process_results = lambda x, y: {metric_name: get_metric(metric_name)}
    #         self.aggregation = lambda: {
    #             metric_name: get_metric_aggregation(metric_name)
    #         }
    #     self._config["metric_list"] = [{"metric": metric_name}]
    #     self._config["process_results"] = "process_results"

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        self.fewshot_rnd = random.Random(seed)
        if hasattr(self, "sampler"):
            self.sampler.set_rnd(seed)

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
        # TODO: this should only return the overrides applied to a non-YAML task's configuration.
        # (num_fewshot)
        return self.config.to_dict()

    @staticmethod
    def process_doc(doc: dict, fn: Callable) -> dict:
        """
        Override this to process (detokenize, strip, replace, etc.) individual
        documents. This can be used in a map over documents of a data split.
        E.g. `map(self._process_doc, self.dataset["validation"])`

        :return: dict
            The processed version of the specified `doc`.
        """
        return doc


class MultipleChoiceTask(Task):
    OUTPUT_TYPE = Literal["loglikelihood"]

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance]:
        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
        if apply_chat_template:
            target_delimiter = (
                self.config.target_delimiter
                if self.config.gen_prefix
                and not ends_with_whitespace(self.config.gen_prefix)
                else ""
            )

        if self._multiple_inputs:
            # If there are multiple inputs, choices are placed in the ctx
            assert isinstance(ctx, list) and isinstance(ctx[0], str), (
                "For multiple input tasks, ctx should be a list of strings"
            )
            assert len(choices) == 1, (
                "For multiple input tasks, there should only be one choice"
            )
            arguments = self.construct_multiple_input_instances(
                context=ctx, choices=choices, target_delimiter=target_delimiter
            )
            # cont = self.doc_to_target(doc)
            # arguments = [(context, f"{target_delimiter}{cont}") for context in ctx]
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
            aux_arguments = self.build_mutual_info(
                context="", choices=choices, target_delimiter=target_delimiter
            )

            arguments.extend(aux_arguments)

        target = self._normalize_target(doc)

        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                idx=i,
                target=target,
                **kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

        return request_list

    @staticmethod
    def build_mutual_info(
        *, context="", choices: list[str], target_delimiter: str
    ) -> list[tuple[str, str]]:
        assert choices is not None and target_delimiter is not None, (
            "choices and target_delimiter must be provided to create acc_mutual_info auxiliary arguments"
        )
        return [(context, f"{target_delimiter}{choice}") for choice in choices]

    def construct_messages(
        self,
        ctx: list[dict[str, Any]],
        choices: list[str],
        target_delimiter: str,
    ) -> list[Instance]:
        if self.config.gen_prefix:
            last_message = ctx[-1]
            _k = "content" if "content" in last_message else "text"
            last_message[_k] += target_delimiter
        raise

    @staticmethod
    def construct_multiple_input_instances(
        *, context: list[str], choices: list[str], target_delimiter: str
    ):
        return [(cxt, f"{target_delimiter}{choices[0]}") for cxt in choices]

    def _normalize_target(self, doc):
        gold = (
            self.doc_to_target(doc)
            if not self._multiple_inputs
            else self.doc_to_text(doc)
        )
        if self._multiple_targets:
            assert isinstance(gold, list), "gold should be a list for multiple targets"
        return gold

    def process_results(self, doc, results):
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

        if self.multiple_inputs:
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
                ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
            ]
            acc_mutual_info = 1.0 if np.argmax(lls_mutual_info) == gold else 0.0
            result_dict["acc_mutual_info"] = acc_mutual_info


class GenerateTask(Task):
    OUTPUT_TYPE: OutputType = "generate_until"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance]:
        assert self.OUTPUT_TYPE == "generate_until"
        name, doc_id, repeats = (
            metadata.pop("task"),
            metadata.pop("doc_id"),
            metadata.pop("repeats"),
        )  # type:ignore[invalid-argument-type]

        # Filter out chat_template and metadata from kwargs
        instance_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("chat_template", "apply_chat_template")
        }

        arguments = (
            ctx,
            cast("dict[str, Any]", deepcopy(self.config.generation_kwargs)),
        )
        multimodal_arguments = self._build_multimodal_kwargs(doc)

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                additional_args=multimodal_arguments,
                target=None,
                idx=0,
                task_name=name,
                doc_id=doc_id,
                repeats=repeats,
                **instance_kwargs,
            )
        ]

    def construct_requests_messages(
        self,
        doc: dict[str, Any],
        ctx: list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance]:
        raise NotImplementedError

    def process_results(self, doc, results):
        result_dict = {}
        use_metric = list(self._metric_fn_list.keys())

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


class LoglikelihoodTask(Task):
    OUTPUT_TYPE: OutputType = "loglikelihood"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance]:
        # TODO: BACKWARD_COMP
        if self.OUTPUT_TYPE == "loglikelihood":
            arguments = (ctx, self.doc_to_target(doc))

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        results = results[0]
        ll, is_greedy = results
        return {
            **({"perplexity": ll} if "perplexity" in use_metric else {}),
            **({"acc": int(is_greedy)} if "acc" in use_metric else {}),
        }


class LoglikelihoodRollingTask(LoglikelihoodTask):
    OUTPUT_TYPE: OutputType = "loglikelihood_rolling"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance]:
        arguments = (self.doc_to_target(doc),)

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                idx=0,
                **kwargs,
            )
        ]

    def process_results(self, doc, results):
        (loglikelihood,) = results
        # TODO: BACKWARD_COMP
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
