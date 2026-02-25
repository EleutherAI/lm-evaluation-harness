from __future__ import annotations

import abc
import logging
import random
import re
from functools import cached_property, partial
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

from tqdm import tqdm

from lm_eval import utils
from lm_eval.api import samplers
from lm_eval.api.utils import (
    Message,
    _build_cache_key,
    group_by_doc_id,
    load_dataset_splits,
    maybe_delimit,
    multiturn_to_singleturn,
    normalize_to_list,
    random_task_id,
    requires_delimiter,
)
from lm_eval.caching.cache import load_from_cache, save_to_cache
from lm_eval.config.task import TaskConfig
from lm_eval.config.utils import (
    _coerce_list,
    _coerce_target,
    process_field,
)
from lm_eval.scorers import ScoredDoc, Scorer, build_scorer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from lm_eval._types import OutputType
    from lm_eval.api._types import ChatTemplate, Dataset, DataSplit, Doc
    from lm_eval.api.instance import AdditionalArgs, GenInstance, Instance, LLInstance
    from lm_eval.config.task import FewshotConfig

eval_logger = logging.getLogger(__name__)


class Task:
    """A task represents an entire benchmark, including its dataset, problems,
    answers, and evaluation methods. See BoolQ for a simple example implementation

    A `doc` can be any python object that represents one instance of evaluation.
    This is usually a dictionary e.g.
        {"question": ..., "answer": ...}
    """

    VERSION: str = "Yaml"  # todo fix version
    OUTPUT_TYPE: OutputType | Literal["multiple_choice"] | None = None
    DATASET_PATH: str | None = None
    DATASET_NAME: str | None = None
    MULTIMODAL: bool = False
    _registry = {}

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
            config = TaskConfig(**config)  # type:ignore[invalid-argument-type]

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
            config if isinstance(config, TaskConfig) else TaskConfig(**config)
        )
        self.task = self._config.task
        self.VERSION = self.config.metadata.get("version", self.VERSION)
        assert self.task is not None
        self.OUTPUT_TYPE = self._config.output_type or self.OUTPUT_TYPE or None
        assert self.OUTPUT_TYPE, "output_type must be set in TaskConfig or subclass"
        self._dataset_name = self._config.dataset_name or self.DATASET_NAME
        self._dataset_path = self._config.dataset_path or self.DATASET_PATH
        self._fewshot_cfg: FewshotConfig = cast(
            "FewshotConfig", self._config.fewshot_config
        )  # normalized by

        self._multiple_inputs = self._config.multiple_inputs
        self._multiple_targets = self.config.multiple_targets
        self._multimodal = (
            bool(self.config.doc_to_audio or self.config.doc_to_image)
            or self.MULTIMODAL
        )

        # lazy load dataset
        self._dataset: Dataset | None = None
        # resolve sampler class, does not need dataset access
        self._sampler_cls: type[samplers.ContextSampler] = self._resolve_sampler_cls()
        # fewshot seed is None by default, so sampler will use random seed.
        self._fewshot_seed: int | None = None
        self._instances = None

        self._scorers: list[Scorer] = self._build_scorers()

    def _resolve_sampler_cls(self) -> type[samplers.ContextSampler]:
        """Resolve the sampler class from config (no dataset access needed)."""
        config_sampler: str | type[samplers.ContextSampler] = (
            self._fewshot_cfg.sampler if self._fewshot_cfg else "default"
        )
        if isinstance(config_sampler, str):
            return samplers.get_sampler(config_sampler)
        elif issubclass(config_sampler, samplers.ContextSampler):
            return config_sampler
        else:
            raise TypeError(
                f"fewshot_config.sampler should be a string or subclass of ContextSampler, "
                f"not {type(config_sampler)}"
            )

    @cached_property
    def sampler(self) -> samplers.ContextSampler:
        """Lazily create the fewshot sampler (triggers dataset download on first access)."""
        fewshot_docs = self.fewshot_docs()
        docs = list(fewshot_docs) if fewshot_docs is not None else []
        return self._sampler_cls(docs, rnd=self._fewshot_seed)

    def _build_scorers(self) -> list[Scorer]:
        """Build scorers from filter_list config, or a default scorer."""
        from lm_eval.api.metrics import Metric
        from lm_eval.api.registry import DEFAULT_METRIC_REGISTRY

        if self.config.metric_list:
            global_metrics = [
                Metric.from_dict({**m_cfg}) for m_cfg in self.config.metric_list
            ]
        else:
            global_metrics = [
                Metric.from_dict({"metric": metric_name})
                for metric_name in DEFAULT_METRIC_REGISTRY.get(self.OUTPUT_TYPE, [])
            ]

        context = self._build_scorer_context()

        if self.config.filter_list:
            scorers = [
                build_scorer(
                    cfg={**cfg},
                    global_metrics=global_metrics,
                    output_type=self.OUTPUT_TYPE,
                    scorer_type=self.config.scorer,
                )
                for cfg in self.config.filter_list
            ]
        else:
            scorers = [
                build_scorer(
                    global_metrics=global_metrics,
                    output_type=self.OUTPUT_TYPE,
                    scorer_type=self.config.scorer,
                )
            ]

        for s in scorers:
            s.context = context
        return scorers

    def _build_scorer_context(self) -> dict[str, Any]:
        """Collect task-level runtime config to forward to metric functions.

        Values here are passed as extra kwargs to every ``Metric.compute``
        call.  ``filter_kwargs`` ensures only metrics whose function
        signature accepts a given key will receive it.
        """
        ctx: dict[str, Any] = {}
        if self._multiple_targets:
            ctx["multiple_targets"] = True
        return ctx

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
                "%s: Custom kwargs can be passed to `--metadata` in console (as json string) or to the TaskManager."
                "\nFor example --metadata='{\"max_seq_lengths\":[4096, 8192]}'. For details see task Readme.",
                self._config.task,
            )
            self._dataset = df(**(self._config.dataset_kwargs | self._config.metadata))
        else:
            assert self._dataset_path is not None, (
                "dataset_path must be set in TaskConfig or class attribute"
            )
            self._dataset = load_dataset_splits(
                path=self._dataset_path,
                name=self._dataset_name,
                split=[
                    self.config.training_split,
                    self.config.validation_split,
                    self.config.test_split,
                    self.config.fewshot_split,
                ],
                **self.config.dataset_kwargs,
            )

    @property
    def dataset(self) -> Dataset:
        """Lazily load and return the dataset."""
        if self._dataset is None:
            self.download(self.config.dataset_kwargs)
        return self._dataset  # type: ignore[return-value]

    @dataset.setter
    def dataset(self, value):
        self._dataset = value  # plain write

    @property
    def config(self) -> TaskConfig:
        """Returns the TaskConfig associated with this class."""
        return self._config

    def has_training_docs(self) -> bool:
        return self.config.training_split is not None

    def has_validation_docs(self) -> bool:
        return self.config.validation_split is not None

    def has_test_docs(self) -> bool:
        return self.config.test_split is not None

    def _get_split_docs(self, split: str | None) -> DataSplit | None:
        """Return the docs for a given split, applying process_docs if configured."""
        if split is None:
            return None
        docs = self.dataset[split]
        if self.config.process_docs is not None:
            return self.config.process_docs(docs)
        return docs

    def training_docs(self) -> DataSplit | None:
        return self._get_split_docs(self.config.training_split)

    def validation_docs(self) -> DataSplit | None:
        return self._get_split_docs(self.config.validation_split)

    def test_docs(self) -> DataSplit | None:
        return self._get_split_docs(self.config.test_split)

    def fewshot_docs(self) -> DataSplit | None:
        if (_df := self._fewshot_cfg.get_docs(self.dataset)) is not None:
            self._fewshot_docs = list(_df)
            return _df

        if (_shots := self._config.num_fewshot) is not None and _shots > 0:
            eval_logger.warning(
                "[Task: %s] num_fewshot > 0 but fewshot_split is None. "
                "using preconfigured rule.",
                self._config.task,
            )
            # Try splits in priority order
            _df = self.training_docs() or self.validation_docs()
            if _df is not None:
                self._fewshot_docs = list(_df)
                return self._fewshot_docs

            # Fallback to test split
            eval_logger.warning(
                "[Task: %s] has_training_docs and has_validation_docs are False"
                ", using test_docs as fewshot_docs but this is not recommended.",
                self._config.task,
            )
            if (_df := self.test_docs()) is not None:
                self._fewshot_docs = list(_df)
                return self._fewshot_docs

            self._fewshot_docs = []
            return self._fewshot_docs

    @property
    def eval_docs(self) -> DataSplit:
        _df = self.test_docs() or self.validation_docs()
        if _df is None:
            raise ValueError(
                f"Task dataset (path={self.DATASET_PATH}, name={self.DATASET_NAME}) must have validation or test docs!"
            )
        return _df

    def get_docs(self, subset: str) -> DataSplit | None:
        split = getattr(self.config, subset, None)
        return self._get_split_docs(split)

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
                "%s: Evaluating on %s examples",
                self.config.task,
                len(samples),
            )
            sample_set = set(samples)
            return utils.create_iterator(
                ((i, x) for i, x in enumerate(self.eval_docs) if i in sample_set),
                rank=int(rank),
                limit=None,
                world_size=int(world_size),
            )

        limit = int(limit) if limit else None
        return utils.create_iterator(
            enumerate(self.eval_docs),
            rank=int(rank),
            limit=limit,
            world_size=int(world_size),
        )

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
        messages: list[Message] = []
        chat_template = (
            partial(chat_template, add_generation_prompt=not gen_prefix)
            if chat_template
            else None
        )
        description = self._resolve_field(doc, self.config.description) or ""
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
                # in most cases we expect q to be a string, except for multiple-input:
                # q: list[str], c: list[str] len 1, a: int index into q
                if isinstance(q, list):
                    assert isinstance(a, int), (
                        "Multiple-input fewshot examples require integer answer keys to index into the question list"
                    )
                    q = q[a]
                    a = 0  # choices are a list of len 1.
                _gen_prefix = self._resolve_field(doc, self._fewshot_cfg.gen_prefix)
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
            # fewshot delimiter used to separate q and gen_prefix
            tgt_delim=self.config.target_delimiter,
            few_delim="",
        )
        if apply_chat_template and chat_template:
            res = (
                [m.to_dict() for m in messages]
                if fewshot_as_multiturn
                else multiturn_to_singleturn(messages)
            )
            res: list[dict[str, str]] | str = chat_template(res)
        else:
            res: str = "".join(m.to_text() for m in messages)

        return res

    def build_qa_turn(
        self,
        *,
        q: str | None,
        c: list[str] | None = None,
        a: str | int | list[str] | list[int] | None = None,
        gen_prefix: str | None = None,
        tgt_delim=" ",
        few_delim="\n\n",
    ) -> list[Message]:
        r"""Build a single Q&A turn as a list of Messages.

        Constructs a user message from the question, and optionally an assistant
        message with the answer (to construct few-shots). Returns intermediate Messages that are assembled
        by ``fewshot_context`` into the final prompt.

        Args:
            q (str): The question or context text.
            c (list[str] | None): List of answer choices for multiple-choice tasks.
                When provided with an integer `a`, indexes into this list to get the answer.
            a (str | int | list[str] | list[int] | None): The answer; only used
                when building few-shot examples. Can be:
                - str: literal answer text
                - int: index into `c`
                - list[str]: multiple valid targets (first used for prompt)
                - list[int]: multiple valid choice indices (first used for prompt)
            gen_prefix (str | None): A prefix to prepend to generated text (e.g., "Answer:").
            tgt_delim (str): Delimiter between question and answer (default: " ").
            few_delim (str): Suffix on the assistant message; acts as separator
                between few-shot turns in plain-text rendering (default: "\n\n").

        Returns:
            list[Message]: [user_msg] or [user_msg, assistant_msg] depending on
                whether an answer or gen_prefix is provided.
        """
        assert isinstance(q, str), (
            f"Expected q to be str, got {type(q).__name__}: {q!r}"
        )
        # Check if answer is provided (handle a=0 as valid answer index)
        has_answer = a is not None and a != ""

        # Determine the delimiter after the user question:
        # - With an answer and no gen_prefix: use tgt_delim (e.g., " ")
        # - With gen_prefix that needs spacing: use tgt_delim
        # - Otherwise: no delimiter
        if (has_answer and not gen_prefix) or (
            gen_prefix and requires_delimiter(q, gen_prefix)
        ):
            _tgt_delim = tgt_delim
        else:
            _tgt_delim = ""

        msgs: list[Message] = [Message("user", q, _tgt_delim)]

        # normalize answer to str
        if has_answer:
            # fmt: off
            match a:
                case str(): answer_text = a
                case int() if c: answer_text = c[a]
                # if a is a list, then we are dealing with multiple-target. Either list[str] or list[int]
                case [str() as first, *_]: answer_text: str = first
                case [int() as first, *_]:
                    assert c is not None, "Answer is an index but no choices provided!"
                    answer_text: str = c[first]
                case _: raise ValueError(f"Unexpected answer type: {a!r}")
            # fmt: on

            # Currently, we always delimit gen_prefex and answer with space if delimiter required.
            answer_text = maybe_delimit(gen_prefix, answer_text, delimiter=" ")
            msgs.append(Message("assistant", answer_text, few_delim))
        elif gen_prefix:
            # For gen-prefix, the delimiter is added in construct_requests
            # when creating the continuation (not for generation tasks).
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
        prev_context = prev_context or []
        results = []
        for ctx in q:
            messages = prev_context + self.build_qa_turn(
                q=ctx, gen_prefix=gen_prefix, tgt_delim=""
            )
            if chat_template:
                formatted = (
                    [m.to_dict() for m in messages]
                    if fewshot_as_multiturn
                    else multiturn_to_singleturn(messages)
                )
                results.append(chat_template(formatted))
            else:
                results.append("".join(m.to_text() for m in messages))
        return results

    @abc.abstractmethod
    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, Any]],
        *,
        doc_id: int,
        metadata: dict[str, Any] | None = None,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[GenInstance] | list[LLInstance] | None:
        """Convert a doc and its prompt context into Instance objects for the LM.

        Called by ``build_all_requests`` after ``fewshot_context`` has produced
        the prompt. Each subclass maps the prompt into the request format its
        output type requires (loglikelihood pairs, generation args, etc.).

        Args:
            doc: The evaluation document from the dataset split.
            ctx: The prompt produced by ``fewshot_context``. Shape depends on
                rendering mode:
                - str: plain-text prompt
                - list[str]: one prompt per input (multiple-input tasks)
                - list[dict]: chat-formatted message list
            doc_id: Index of the document within the evaluation split.
            metadata: Per-instance metadata forwarded to the Instance.
            apply_chat_template: Whether a chat template was applied.
            chat_template: The chat template callable, if any.

        Returns:
            A list of Instances to send to the LM, or None to skip this doc.
        """
        ...

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
        """Build all Instance objects for this task and store them in ``self._instances``.

        For each document in the evaluation split this method:
        1. Builds the prompt via ``fewshot_context``.
        2. Converts it to Instance(s) via ``construct_requests``.
        3. Optionally loads/saves results from a request cache.

        Args:
            limit: Maximum number of documents to evaluate (None = all).
            samples: Explicit list of document indices to evaluate.
            rank: Worker rank for distributed evaluation.
            world_size: Total number of workers.
            cache_requests: Whether to load/save instances from cache.
            rewrite_requests_cache: Force-rebuild the cache even if it exists.
            system_instruction: System prompt prepended to every context.
            apply_chat_template: Whether to render prompts through a chat template.
            fewshot_as_multiturn: Keep few-shot examples as separate chat turns
                instead of collapsing them into a single user message.
            chat_template: The chat template callable.
            tokenizer_name: Included in the cache key to distinguish tokenizers.

        Returns:
            Flat list of Instances, also stored in ``self._instances``.
        """
        cache_key = _build_cache_key(
            self._config.task,
            self.config.num_fewshot,
            rank,
            world_size,
            apply_chat_template,
            fewshot_as_multiturn,
            system_instruction,
            tokenizer_name,
        )
        cached = load_from_cache(file_name=cache_key, cache=cache_requests)

        if cache_requests and cached and not rewrite_requests_cache:
            grouped = cached[:limit]
        else:
            # When caching a miss/rewrite, build ALL docs so the cache is
            # complete; then slice to the requested limit afterwards.
            should_build_all = (
                cache_requests
                and (not cached or rewrite_requests_cache)
                and limit is not None
            )
            build_limit = None if should_build_all else limit

            eval_logger.info(
                "Building contexts for %s on rank %s...",
                self.config.task,
                rank,
            )

            grouped: list[list[Instance]] = []
            doc_id_docs = list(
                self.doc_iterator(
                    rank=rank, limit=build_limit, samples=samples, world_size=world_size
                )
            )

            for doc_id, doc in tqdm(doc_id_docs, total=len(doc_id_docs), delay=5):
                fewshot_ctx = self.fewshot_context(
                    doc,
                    num_fewshot=0
                    if self.config.num_fewshot is None
                    else self.config.num_fewshot,
                    system_instruction=system_instruction,
                    apply_chat_template=apply_chat_template,
                    fewshot_as_multiturn=fewshot_as_multiturn,
                    chat_template=chat_template,
                    gen_prefix=self._resolve_field(doc, self.config.gen_prefix),
                )

                # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
                inst = self.construct_requests(
                    doc=doc,
                    ctx=fewshot_ctx,
                    doc_id=doc_id,
                    apply_chat_template=apply_chat_template,
                    chat_template=chat_template,
                )
                if inst is None:
                    eval_logger.info("Skipping doc_id=%s.", doc_id)
                    continue
                if not isinstance(inst, list):
                    inst = [inst]

                grouped.append(inst)

            if cache_requests and (not cached or rewrite_requests_cache):
                save_to_cache(file_name=cache_key, obj=grouped)

            grouped = grouped[:limit]

        flattened = [inst for group in grouped for inst in group]

        if not flattened:
            raise ValueError("task.build_requests() did not find any docs!")

        self._instances = flattened
        return flattened

    @property
    def instances(self) -> list[Instance]:
        """After calling `task.build_all_requests()`, tasks
        maintain a list of the dataset instances which will be evaluated.
        """
        return self._instances or []

    ### Doc to Text/Target/Choice/Multimodal Parsing Methods ##

    def doc_to_text(
        self,
        doc: Doc,
        doc_to_text: Callable[[Doc], str | list[str]] | str | None = None,
    ) -> str | list[str] | None:
        doc_to_text = (
            doc_to_text if doc_to_text is not None else self.config.doc_to_text
        )
        return process_field(doc, doc_to_text)

    def doc_to_choice(
        self,
        doc: Doc,
        doc_to_choice: Callable[[Doc], list[str]] | str | list[str] | None = None,
    ) -> list[str] | None:
        doc_to_choice = (
            doc_to_choice if doc_to_choice is not None else self.config.doc_to_choice
        )
        choices = _coerce_list(process_field(doc, doc_to_choice))
        if choices is not None and not isinstance(choices, list):
            eval_logger.warning(
                "doc_to_choice must return a list, got %s: %r. Skipping ...",
                type(choices).__name__,
                choices,
            )
            return None
        return choices

    def doc_to_target(
        self,
        doc: Doc,
        doc_to_target: Callable[[Doc], str | int | list[int] | list[str]]
        | str
        | None = None,
    ) -> str | int | list[str] | list[int] | None:
        doc_to_target = (
            doc_to_target if doc_to_target is not None else self.config.doc_to_target
        )
        y = process_field(doc, doc_to_target)
        return _coerce_target(y, parse_list=self._multiple_targets is True)

    def doc_to_image(self, doc: Any, doc_to_image=None) -> int | str | list | None:
        return process_field(doc, doc_to_image or self.config.doc_to_image)

    def doc_to_audio(self, doc: Any, doc_to_audio=None) -> int | str | list | None:
        return process_field(doc, doc_to_audio or self.config.doc_to_audio)

    @cached_property
    def is_multimodal(self):
        return (
            self.config.doc_to_image is not None or self.config.doc_to_audio is not None
        )

    def _build_multimodal_kwargs(self, doc) -> AdditionalArgs | None:
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

        instances = group_by_doc_id(self._instances)

        for scorer in self._scorers:
            pr_results = self._try_process_results(instances, scorer.name)
            if pr_results is not None:
                scorer.set_results(pr_results)
            else:
                scored_docs = scorer.score_instances(instances)
                scorer.set_results(scored_docs)

    def _try_process_results(
        self,
        instances: dict[int, list[Instance]],
        filter_key: str,
    ) -> dict[int, ScoredDoc] | None:
        """Run custom process_results path for all docs.

        Returns ``{doc_id: ScoredDoc}`` if ``process_results`` produces
        results, or ``None`` if it returns ``None`` for the first document
        (signaling that scoring should fall through to the scorer pipeline).

        Raises ``ValueError`` if ``process_results`` returns ``None``
        inconsistently (non-``None`` for some docs, ``None`` for others).
        """

        accumulator: dict[int, ScoredDoc] = {}

        for doc_id, doc_instances in instances.items():
            # we need to remove one level of nesting:
            metrics = self.process_results(
                doc_instances[0].doc,
                list(
                    chain.from_iterable(
                        [req.filtered_resps[filter_key] for req in doc_instances]
                    )
                ),
            )
            if metrics is None:
                if accumulator:
                    raise ValueError(
                        f"process_results() returned None for doc_id={doc_id} after "
                        f"returning results for {len(accumulator)} earlier documents. "
                        f"process_results() must consistently return either a dict or "
                        f"None for all documents."
                    )
                return None

            scores = cast(
                "dict[str, list[float]]", cast("object", normalize_to_list(metrics))
            )
            scored_doc = ScoredDoc(
                doc_id=doc_id, reference=doc_instances[0].target, scores=scores
            )
            accumulator[doc_id] = scored_doc

        return dict(accumulator) or None

    def process_results(
        self, doc: dict[str, Any], results: list[Any]
    ) -> dict[str, list[Any]] | None:
        if callable(self.config.process_results) and not isinstance(
            self.config.process_results, str
        ):
            return self.config.process_results(doc, results)
        return None

    def aggregate(
        self, bootstrap_iters: int | None = 100000
    ) -> tuple[dict[str, Any], int]:
        """Aggregate all scorers' reduced results.

        Returns (agg_dict, sample_len) where agg_dict has "metric,scorer" string keys.
        This is the only place where string keys are produced.

        Legacy Python tasks that override ``aggregation()`` get their custom
        functions forwarded to each scorer so that corpus-level metrics
        (e.g. SQuAD v2, SCROLLS) are aggregated correctly instead of
        falling back to ``mean``.
        """
        # Detect subclass override of aggregation()
        custom_agg = (
            self.aggregation()
            if type(self).aggregation is not Task.aggregation
            else None
        )

        agg_metrics: dict[str, Any] = {}
        sample_len = 0
        for scorer in self._scorers:
            result, count = scorer.aggregate(
                bootstrap_iters=bootstrap_iters,
                aggregation_overrides=custom_agg,
            )
            agg_metrics.update(result)
            sample_len = max(sample_len, count)
        return agg_metrics, sample_len

    def export_raw_metrics(self) -> dict[str, dict[str, list[Any]]]:
        """Export reduced results from all scorers for distributed gathering.

        Returns {scorer_name: {metric_name: [per_doc_values]}}.
        """
        exported: dict[str, dict[str, list[Any]]] = {}
        for scorer in self._scorers:
            metrics = scorer.export_reduced()
            if metrics:
                exported[scorer.name] = metrics
        return exported

    def import_raw_metrics(self, data: dict[str, dict[str, list]]) -> None:
        """Import merged results into scorers (after distributed gather).

        Rebuilds scored docs from flat metric lists so that
        ``scorer.aggregate()`` works.
        """
        for scorer in self._scorers:
            if scorer.name not in data:
                continue
            scorer.import_reduced(data[scorer.name])

    ## Some public methods and utilities ##

    @property
    def scorers(self) -> list[Scorer]:
        """Public access to the scorer pipeline."""
        return self._scorers

    @property
    def task_name(self) -> str:
        return getattr(self.config, "task", random_task_id())

    @cached_property
    def id(self):
        from random import Random

        return Random(f"{self.task_name}").randint(0, 2**32)

    def aggregation(self) -> dict[str, Callable[[list[Any]], Any]]:
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
        from lm_eval.api.metrics import Metric

        metric = Metric.from_dict({"metric": metric_name})
        self._scorers = [
            build_scorer(global_metrics=[metric], output_type=self.OUTPUT_TYPE)
        ]

    @staticmethod
    def _resolve_field(doc: dict[str, Any], field: str | None = None) -> str | None:
        return cast("str | None", cast("object", process_field(doc, field)))

    def set_fewshot_seed(self, seed: int | None = None) -> None:
        self.fewshot_rnd = random.Random(seed)
        self._fewshot_seed = seed
        # If sampler already materialized, update it directly
        if "sampler" in self.__dict__:
            self.sampler.set_rnd(seed)

    def dump_config(self) -> dict:
        """Returns the config as a dictionary."""
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
