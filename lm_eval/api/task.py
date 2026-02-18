from __future__ import annotations

import abc
import ast
import logging
import random
import re
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from distlib.util import cached_property
from tqdm import tqdm
from typing_extensions import TypedDict

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.instance import AdditionalArgs, Instance
from lm_eval.api.registry import DEFAULT_METRIC_REGISTRY
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
from lm_eval.config.utils import process_field
from lm_eval.scorers import Scorer
from lm_eval.utils import normalize_to_list


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import datasets

    from lm_eval._types import ChatTemplate, OutputType
    from lm_eval.config.task import FewshotConfig

eval_logger = logging.getLogger(__name__)


class METADATA(TypedDict, total=True, extra_items=Any):
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
        self._multiple_targets = self.config.multiple_targets
        self._multimodal = self.MULTIMODAL or bool(
            self.config.doc_to_audio or self.config.doc_to_image
        )

        # Must be set before fewshot_docs() which may trigger dataset access
        self._dataset = None
        self._instances = None

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

        self._scorers: list[Scorer] = self._build_scorers()

    def _build_global_metrics(self) -> list:
        """Convert config.metric_list (or output_type defaults) into list[Metric]."""
        from lm_eval.config.metric import Metric

        metrics: list[Metric] = []
        if self.config.metric_list is not None:
            for m_cfg in self.config.metric_list:
                metrics.append(Metric.from_dict(m_cfg))
        else:
            for metric_name in DEFAULT_METRIC_REGISTRY.get(self.OUTPUT_TYPE, []):
                metrics.append(Metric.from_dict({"metric": metric_name}))
        return metrics

    def _build_scorers(self) -> list[Scorer]:
        """Build scorers from filter_list config, or a default scorer."""
        global_metrics = self._build_global_metrics()
        output_type = self.OUTPUT_TYPE

        if self.config.filter_list is not None:
            return [
                Scorer.from_dict(
                    filter_config,
                    global_metrics=global_metrics,
                    output_type=output_type,
                )
                for filter_config in self.config.filter_list
            ]

        return [Scorer.default_scorer(global_metrics, output_type=output_type)]

    def _has_metric(self, metric_name: str) -> bool:
        """Check if any scorer contains a metric with the given name."""
        return any(
            m.name == metric_name
            for scorer in self._scorers
            for m in (scorer.metrics or [])
        )

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

    def has_training_docs(self) -> bool:
        return self.config.training_split is not None

    def has_validation_docs(self) -> bool:
        return self.config.validation_split is not None

    def has_test_docs(self) -> bool:
        return self.config.test_split is not None

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
                ((i, x) for i, x in enumerate(self.eval_docs) if i in samples),
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
        assert isinstance(q, str), (
            f"Expected doc_to_text to be a string, got {type(q)}: {q}"
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
            res: str = "".join(m.to_text() for m in messages)

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
    ) -> list[Instance] | Instance | None: ...

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
            if inst is None:
                eval_logger.info(f"Skipping {doc_id=}.")
                continue
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

    def doc_to_text(self, doc, doc_to_text=None) -> str | list[str] | None:
        doc_to_text = (
            doc_to_text if doc_to_text is not None else self.config.doc_to_text
        )
        y = process_field(doc, doc_to_text, digits=True, lists=False, default=None)
        return y

    def doc_to_choice(self, doc, doc_to_choice=None) -> list[str] | None:
        doc_to_choice = (
            doc_to_choice if doc_to_choice is not None else self.config.doc_to_choice
        )
        y = process_field(doc, doc_to_choice, digits=True, lists=True, default=[])
        return y

    def doc_to_target(
        self, doc, doc_to_target=None
    ) -> str | int | list[str] | list[int] | None:
        doc_to_target = (
            doc_to_target if doc_to_target is not None else self.config.doc_to_target
        )
        y = process_field(doc, doc_to_target, digits=True, lists=True, default=None)
        return y

    def doc_to_prefix(self, doc):
        if (gen_prefix := self.config.gen_prefix) is not None:
            if gen_prefix in doc:
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
            if doc_to_image in doc:
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
            if doc_to_audio in doc:
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

    def apply_filters(self) -> None:
        """Apply filter ensembles from each scorer to instances."""
        if not self._instances:
            return
        for scorer in self._scorers:
            scorer.apply_filter(self._instances)

    def process_instances(self) -> None:
        """Apply filters, score instances, reduce — all stored on Scorers.

        For each scorer, tries the legacy ``process_results`` path first
        (YAML ``!function`` or Python subclass override).  Falls through to
        ``scorer.score_instances()`` only when ``process_results`` returns
        ``None``.
        """
        if not self._scorers:
            return

        self.apply_filters()

        instances = self._sort_instances(self._instances)

        # try process results first
        for scorer in self._scorers:
            pr_results = self._try_process_results(instances, scorer.name)
            if pr_results is not None:
                # Legacy path: process_results returns pre-scored values.
                # normalize_to_list wraps scalars in single-element lists,
                # giving _metric_results shape: {"acc": [[0], [1], [1]]}
                scorer._metric_results = defaultdict(list, pr_results)
            else:
                scorer.score_instances(instances)
            scorer.reduce()  # _metric_results → _reduced_results

    def _try_process_results(
        self,
        instances: dict[int, list[Instance]],
        filter_key: str,
    ) -> dict[str, list[list[Any]]] | None:
        """Try the legacy process_results path for all docs.

        Returns ``{metric_name: [per_doc_values]}`` if ``process_results``
        returns a non-None dict for the first document, otherwise ``None``.
        """
        from collections import defaultdict

        accumulator = defaultdict(list)

        for doc_id, doc_instances in instances.items():
            filtered_resps = [req.filtered_resps[filter_key] for req in doc_instances]
            doc = doc_instances[0].doc
            metrics = self.process_results(doc, filtered_resps)

            if metrics is None:
                return None

            metrics = normalize_to_list(metrics)

            for metric_name, value in metrics.items():
                accumulator[metric_name].append(value)

        return dict(accumulator) if accumulator else None

    def process_results(self, doc, results) -> dict[str, list[Any]] | None:
        if callable(self.config.process_results):
            return self.config.process_results(doc, results)
        return None

    def aggregate(
        self, bootstrap_iters: int | None = 100000
    ) -> tuple[dict[str, Any], int]:
        """Aggregate all scorers' reduced results.

        Returns (agg_dict, sample_len) where agg_dict has "metric,scorer" string keys.
        This is the only place where string keys are produced.
        """
        agg_metrics: dict[str, Any] = {}
        sample_len = 0
        for scorer in self._scorers:
            result, count = scorer.aggregate(bootstrap_iters=bootstrap_iters)
            agg_metrics.update(result)
            sample_len = max(sample_len, count)
        return agg_metrics, sample_len

    def export_raw_metrics(self) -> dict[str, dict[str, list]]:
        """Export reduced results from all scorers for distributed gathering.

        Returns {scorer_name: {metric_name: [per_doc_values]}}.
        """
        return {
            scorer.name: dict(scorer._reduced_results)
            for scorer in self._scorers
            if scorer._reduced_results
        }

    def import_raw_metrics(self, data: dict[str, dict[str, list]]) -> None:
        """Import merged results into scorers (after distributed gather)."""
        for scorer in self._scorers:
            if scorer.name in data:
                scorer._reduced_results = data[scorer.name]

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
        return {
            m.name: m.aggregation
            for scorer in self._scorers
            for m in (scorer.metrics or [])
            if m.aggregation
        }

    def higher_is_better(self) -> dict[str, bool]:
        return {
            k: v for scorer in self._scorers for k, v in scorer.higher_is_better.items()
        }

    def get_config(self, key: str) -> Any:
        return getattr(self._config, key, None)

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
        """Override the default metrics with a single named metric.

        Rebuilds the scorer pipeline so that only *metric_name* is computed.
        Used by the evaluator for ``predict_only`` mode (metric="bypass").
        """
        from lm_eval.config.metric import Metric

        metric = Metric.from_dict({"metric": metric_name})
        self._scorers = [Scorer.default_scorer([metric], output_type=self.OUTPUT_TYPE)]

    @staticmethod
    def resolve_field(doc: dict[str, Any], field: str | None = None):
        if field:
            return doc[field] if field in doc else utils.apply_template(field, doc)

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
    OUTPUT_TYPE: OutputType = "multiple_choice"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        metadata: METADATA,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[Instance] | None:
        name = metadata.get("task", self.task_name)
        doc_id = metadata.get("doc_id", 0)
        repeats = metadata.get("repeats", 1)

        choices = self.doc_to_choice(doc)
        if not choices:
            eval_logger.warning(
                f"No choices found for doc:\n\n{doc}\n\nSkipping this instance."
            )
            return None
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

        # If any scorer uses acc_mutual_info, we need unconditional loglikelihoods.
        # This computes log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
        # by appending ("", continuation) pairs for each choice.
        # NOTE: this will at most ~2x runtime.
        if self._has_metric("acc_mutual_info"):
            aux_arguments = self.build_mutual_info(
                context="", choices=choices, target_delimiter=target_delimiter
            )
            arguments.extend(aux_arguments)
            metadata.update({"acc_mutual_info": True})

        target = self.doc_to_target(doc)

        if target is None:
            return None

        num_choices = len(choices)  # noqa: F841
        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                task_name=name,
                idx=i,
                doc_id=doc_id,
                repeats=repeats,
                target=target,
                metadata=metadata,
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

    def doc_to_target(self, doc, doc_to_target=None) -> int | list[int] | None:
        doc_to_target = super().doc_to_target(doc, doc_to_target)
        if isinstance(doc_to_target, int):
            return doc_to_target

        if isinstance(doc_to_target, str):
            choices = self.doc_to_choice(doc)
            if doc_to_target in choices:
                return choices.index(doc_to_target)
            else:
                eval_logger.warning(
                    f"[{self.task}] Target '{doc_to_target}' not found in choices {choices} for doc:\n\n{doc}\n\n"
                )
                return None  # invalid index to indicate error

        if isinstance(doc_to_target, list):
            choices = self.doc_to_choice(doc)
            target_indices = []
            for target in doc_to_target:
                if target in choices:
                    target_indices.append(choices.index(target))
                else:
                    eval_logger.warning(
                        f"Target '{target}' not found in choices {choices} for doc:\n\n{doc}\n\n"
                    )
                    target_indices.append(-100)  # invalid index to indicate error
            return target_indices or None

    def doc_to_text(self, doc, doc_to_text=None) -> str | list[str] | None:
        doc_to_text = super().doc_to_text(doc, doc_to_text)
        if isinstance(doc_to_text, str):
            return doc_to_text
        elif isinstance(doc_to_text, list):
            assert self._multiple_inputs, (
                "doc_to_text should return a single string for non-multiple-input tasks"
            )
            return doc_to_text

    def doc_to_choice(self, doc, doc_to_choice=None) -> list[str] | None:
        choices = super().doc_to_choice(doc, doc_to_choice)
        if choices is not None and not isinstance(choices, list):
            raise ValueError(
                "doc_to_choice should return a list of strings representing the answer choices."
            )
        if self._multiple_inputs:
            assert choices is not None and len(choices) == 1, (
                "For multiple input tasks, doc_to_choice should return a list with a single string representing the answer choice template."
            )
        return choices


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
        multimodal_arguments = (
            self._build_multimodal_kwargs(doc) if self._multimodal else None
        )
        target = self.doc_to_target(doc)

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                additional_args=multimodal_arguments,
                target=target,
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


class LoglikelihoodTask(Task):
    OUTPUT_TYPE: OutputType = "loglikelihood"

    def __init__(self, config: TaskConfig | dict[str, Any], *args, **kwargs):
        super().__init__(config)
        assert self._multiple_inputs is False
        assert self._multiple_targets is False

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
        name = metadata.get("task", self.task_name)
        doc_id = metadata.get("doc_id", 0)
        repeats = metadata.get("repeats", 1)

        # TODO: BACKWARD_COMP
        if self.OUTPUT_TYPE == "loglikelihood":
            cont = self.doc_to_target(doc)
            assert isinstance(cont, str), (
                "For loglikelihood tasks, the argument should be a string representing the continuation to score against the context."
            )
            arguments = (ctx, cont)

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                task_name=name,
                idx=0,
                doc_id=doc_id,
                repeats=repeats,
                **kwargs,
            )
        ]


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
        name = metadata.get("task", self.task_name)
        doc_id = metadata.get("doc_id", 0)
        repeats = metadata.get("repeats", 1)

        arguments = (self.doc_to_target(doc),)

        return [
            Instance(
                request_type=self.OUTPUT_TYPE,
                doc=doc,
                arguments=arguments,
                task_name=name,
                idx=0,
                doc_id=doc_id,
                repeats=repeats,
                **kwargs,
            )
        ]


# Backward compatibility alias
ConfigurableTask = Task
