from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import Required, TypedDict

from lm_eval.defaults import default_gen_kwargs

from .utils import normalize_filter_list, normalize_metric_list


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import Literal

    from lm_eval.api._types import (
        Completion,
        Dataset,
        DataSplit,
        Doc,
        GenKwargs,
        LLOutput,
    )
    from lm_eval.api.instance import OutputType
    from lm_eval.api.metrics import AggregationFn, MetricFn, ReductionFn

    _OutputType = OutputType | Literal["multiple_choice"]


eval_logger = logging.getLogger(__name__)


class MetricConfig(TypedDict, total=False):
    """Configuration for a single metric in ``metric_list``.

    Example:
        ```yaml
        metric_list:
          - metric: exact_match
            aggregation: mean
            higher_is_better: true
            ignore_case: true       # extra kwarg forwarded to the metric fn
        ```
    """

    metric: Required[str | MetricFn]
    """Name of a registered metric (e.g. ``"acc"``, ``"exact_match"``,
    ``"bleu"``) or a callable. See ``lm_eval/api/metrics.py`` for
    built-in metrics."""

    aggregation: str | AggregationFn | None
    """How per-document metric values are combined into a single score
    (e.g. ``"mean"``, ``"median"``). Can be a registered aggregation or a callable.
    Defaults to the metric's registered aggregation if not set."""

    reduction: str | ReductionFn | None
    """How per-instance repeated values are reduced before aggregation
    (e.g. when ``repeats > 1``). Can be a registered reduction or a callable.
    Defaults to the metric's registered reduction if not set."""

    higher_is_better: bool | None
    """Whether a higher metric value indicates better performance.
    Defaults to the metric's registered value if not set."""

    kwargs: dict[str, Any] | None
    """Extra keyword arguments forwarded to the metric function
    (e.g. ``{"ignore_case": true}`` for ``exact_match``) Extreneous fields are also.
    treated as kwargs.
    """


class FilterStep(TypedDict, total=False):
    r"""A single filter step in a pipeline.

    Example:
        ```yaml
        - function: "regex"
          kwargs:
            regex_pattern: "#### (\\\\-?[0-9\\\\.\\\\,]+)"

        - function: "custom"
          kwargs:
            filter_fn: !function my_custom_filter_fn
        ```
    """

    function: Required[str]
    """Name of a registered filter (e.g. ``"regex"``, ``"custom"``,
    ``"majority_vote"``). Custom filters can be registered with
    ``@register_filter``."""

    kwargs: dict[str, Any]
    r"""Keyword arguments passed to the filter function
    (e.g. ``{"regex_pattern": "The answer is (\d+)"}`` for ``"regex"``,
    or ``{"filter_fn": "!function utils.my_filter"}`` for ``"custom"``)."""


class ScorerConfig(TypedDict, total=False):
    """Configuration for a registered scorer.

    A scorer encapsulates the full filter → score → reduce → aggregate
    pipeline. When ``scorer`` is set on a task, scoring is delegated to
    the registered scorer class instead of the default metric pipeline.

    Can be specified as a plain string (just the scorer name) or as a
    dict with ``type`` and optional ``kwargs`` forwarded to the scorer
    constructor.

    Example:
        ```yaml
        # String shorthand — equivalent to {"type": "first_token"}
        scorer: first_token

        # Dict form with custom kwargs forwarded to the scorer class
        scorer:
          type: ai_judge
          kwargs:
            judge_model: claude-sonnet-4-6
        ```

    See ``lm_eval/scorers/`` for built-in scorers.  Custom scorers can
    be registered with ``@register_scorer``.
    """

    type: Required[str]
    """Name of a registered scorer (e.g. ``"first_token"``, ``"regex"``,
    ``"choice_match"``). Resolved via ``lm_eval.api.registry.get_scorer``."""

    kwargs: dict[str, Any]
    """Extra keyword arguments forwarded to the scorer constructor.
    Scorer subclasses declare these as dataclass fields
    (e.g. ``judge_model`` on an ``AIJudgeScorer``)."""


class FilterPipeline(TypedDict, total=False):
    r"""A named filter pipeline with optional per-pipeline metrics.

    Mirrors each element of the ``filter_list`` entries in YAML task configs.

    Example:
        ```yaml
        filter_list:
          - name: "strict-match"
            filter:
              - function: "regex"
                regex_pattern: "#### (\\-?[0-9\\.\\,]+)"
              - function: "take_first"
            metric_list:
                - metric: "exact_match"
          - name: "loose-match"
            ...
        ```
    """

    name: Required[str]
    """Identifier for this pipeline, used as a prefix in result keys
    (e.g. ``"strict-match"``, ``"maj@64"``)."""

    filter: Required[list[FilterStep]]
    """Ordered sequence of filter steps applied to model outputs.
    Steps run in order; each step's output feeds into the next."""

    metric_list: list[MetricConfig]
    """Optional per-pipeline metrics. When set, these metrics are scored
    only on this pipeline's filtered outputs, instead of the task-level
    ``metric_list``."""


@dataclass
class FewshotConfig:
    """Configuration for few-shot example formatting.

    These fields override the parent TaskConfig fields when formatting
    few-shot examples (as opposed to the test example).

    note: num_fewshot is also runtime-dependent, so is not included here.
    """

    sampler: str = "default"
    """Sampling strategy for selecting few-shot examples (e.g. ``"default"``,
    ``"first_n"``). ``"default"`` samples randomly."""

    split: str | None = None
    """Dataset split to draw few-shot examples from. Inherited from
    ``TaskConfig.fewshot_split`` if not set directly. Takes precedence
    over ``samples`` when both are provided."""

    process_docs: Callable[..., list[dict[str, Any]]] | None = None
    """Optional callable to transform the few-shot split before sampling.
    Inherited from ``TaskConfig.process_docs`` if not set."""

    fewshot_indices: list[int] | None = None
    """Explicit list of document indices to use as few-shot examples.
    When set, overrides random sampling with a fixed selection."""

    samples: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None
    """Hardcoded few-shot examples as a list of dicts, or a callable
    returning such a list. Used when examples don't come from a dataset
    split. Ignored if ``split`` is also set."""

    doc_to_text: str | Callable[[Doc], str] | None = None
    """Override ``doc_to_text`` for formatting few-shot examples differently
    from the test example. Inherited from ``TaskConfig.doc_to_text``."""

    doc_to_choice: str | Callable[[Doc], list[str]] | list[str] | None = None
    """Override ``doc_to_choice`` for few-shot examples.
    Inherited from ``TaskConfig.doc_to_choice``."""

    doc_to_target: str | Callable[[Doc], str | int] | None = None
    """Override ``doc_to_target`` for few-shot examples.
    Inherited from ``TaskConfig.doc_to_target``."""

    gen_prefix: str | None = None
    """Override ``gen_prefix`` for few-shot examples.
    Inherited from ``TaskConfig.gen_prefix``."""

    fewshot_delimiter: str | None = None
    """Override the delimiter between few-shot examples.
    Inherited from ``TaskConfig.fewshot_delimiter``."""

    target_delimiter: str | None = None
    """Override the delimiter between prompt and target in few-shot examples.
    Inherited from ``TaskConfig.target_delimiter``."""

    def __post_init__(self):
        if self.split is not None and self.samples is not None:
            eval_logger.warning(
                "Both split and samples are configured; split will take precedence"
            )

    def get_docs(self, dataset: Dataset) -> DataSplit | None:
        if self.split is not None:
            if self.process_docs is not None:
                return self.process_docs(dataset[self.split])
            return dataset[self.split]
        elif self.samples is not None:
            # fmt: off
            match self.samples:
                case list(): return cast("list[dict[str, Any]]", self.samples)
                case fsamples if callable(self.samples): return cast("list[dict[str, Any]]", fsamples())
                case _: raise ValueError(
                        "`fewshot_config['samples']` was incorrectly defined in the configuration. It should either be `list[dict]`, or callable returning this list."
                    ) from None
            # fmt: on

    @classmethod
    def from_dict(
        cls,
        cfg: Mapping[str, Any],
        *,
        # inherited from TaskConfig if not specified
        fewshot_split: str | None = None,
        process_docs: Callable[[Iterable[dict[str, Any]]], list[Doc]] | None = None,
        fewshot_delimiter: str | None = None,
        target_delimiter: str | None = None,
        gen_prefix: str | None = None,
        doc_to_text: str | Callable[[Doc], str | list[str]] | None = None,
        doc_to_choice: str | Callable[[Doc], list[str]] | list[str] | None = None,
        doc_to_target: str
        | Callable[[Doc], str | int | list[int] | list[str]]
        | None = None,
        **overloads,
    ) -> FewshotConfig:
        cfg_dict = {
            "split": fewshot_split,
            "process_docs": process_docs,
            "fewshot_delimiter": fewshot_delimiter,
            "target_delimiter": target_delimiter,
            "gen_prefix": gen_prefix,
            "doc_to_text": doc_to_text,
            "doc_to_choice": doc_to_choice,
            "doc_to_target": doc_to_target,
            **cfg,
            **overloads,
        }
        cfg_dict.setdefault("sampler", "default")
        return cls(**cfg_dict)  # type:ignore[invalid-argument-type]


@dataclass(kw_only=True)
class TaskConfig:
    """Configuration for a single evaluation task.

    Maps 1:1 with the YAML task config files under ``lm_eval/tasks/``.
    Every key in a task YAML corresponds to a field here.

    Example:
        ```yaml
        task: arc_easy @ cloze
        dataset_path: allenai / ai2_arc
        dataset_name: ARC - Easy
        test_split: test
        doc_to_text: "{{question}}"
        doc_to_target: "{{choices.label.index(answerKey)}}"
        doc_to_choice: "{{choices.text}}"
        ```
    """

    # ── Task naming / registry ──────────────────────────────────────────

    task: str
    """Unique task identifier used for registration and CLI selection
    (e.g. ``--tasks arc_easy``). Append ``@<formats>`` to select a prompt
    formats at runtime (e.g. ``"arc_easy@cloze"``)."""

    task_alias: str | None = None
    """Optional display name shown in result tables instead of ``task``."""

    formats: str | dict[str, str] | None = None
    """Prompt formats to apply. Can be a registered formats name (e.g. ``"cloze"``,
    ``"mcq"``), or an inline dict of [FormatConfig][.formats.FormatConfig] fields. When set, the format's
    template overrides ``doc_to_text``, ``doc_to_target``, ``output_type``, etc.
    If None and ``task`` contains ``@``, the suffix is used as the formats name."""

    output_type: _OutputType = "generate_until"
    """The type of model request to construct for each document.

    - ``"generate_until"``: open-ended text generation (default).
    - ``"loglikelihood"``: score the likelihood of a target string.
    - ``"loglikelihood_rolling"``: score a full sequence without a context split.
    - ``"multiple_choice"``: rank answer choices by loglikelihood.
    """

    tag: list[str] = field(default_factory=list)
    """Tags for categorizing this task. Users can select all tasks sharing
    a tag via ``--tasks <tag_name>`` (e.g. ``tag: [math, reasoning]`` lets
    users run ``--tasks math`` to include this task).
    Distinct from explicit ``group`` configs (see ``GroupConfig``)."""

    # ── HF dataset options ──────────────────────────────────────────────

    custom_dataset: Callable[..., Dataset] | None = None
    """A callable that returns a HuggingFace ``DatasetDict``. Should accept
    arbitrary kwargs. Typically set in YAML via ``!function utils.xxx``
    Use this when you need custom loading logic instead of ``datasets.load_dataset``.
    At runtime, receives ``metadata`` (from this config) and ``model_args``
    (if using ``evaluate``) as keyword arguments."""

    dataset_path: str | None = None
    """HuggingFace dataset path passed to ``datasets.load_dataset()``.
    Can be a Hub identifier (e.g. ``"allenai/ai2_arc"``) or a local path."""

    dataset_name: str | None = None
    """HuggingFace dataset config/subset name (e.g. ``"ARC-Easy"``)."""

    dataset_kwargs: dict[str, str | int | float] = field(default_factory=dict)
    """Extra keyword arguments forwarded to ``datasets.load_dataset()``
    (e.g. ``{"data_dir": "path/to/data"}`` or ``{"data_files": "data.json"}``)."""

    training_split: str | None = None
    """Name of the training split in the dataset (e.g. ``"train"``)."""

    validation_split: str | None = None
    """Name of the validation split. Used as the evaluation split when
    ``test_split`` is not set."""

    test_split: str | None = None
    """Name of the test split. When set, this is the primary split evaluated."""

    fewshot_split: str | None = None
    """Name of the split from which few-shot examples are drawn. Passed as
    the default for ``fewshot_config.split``; overridden if ``fewshot_config``
    explicitly sets its own ``split``."""

    # ── Formatting / prompting options ──────────────────────────────────

    process_docs: Callable[..., list[dict[str, Any]]] | None = None
    """A callable applied to a dataset split before evaluation. Use this
    to filter, transform, or resample documents (e.g. renaming columns,
    expanding multi-answer rows). Typically set in YAML via ``!function utils.xxx``"""

    description: str = ""
    """A Jinja2 template or plain string prepended to every prompt. Useful for
    task-level instructions, e.g.
    ``"The following are questions (with answers) about {{subject}}.\n\n"``.
    When a chat template is applied, this is combined with
    ``system_instruction`` and sent as the ``system`` message."""

    doc_to_text: str | Callable[[Doc], str | list[str]] | None = None
    """Converts a document dict into the prompt text shown to the model.
    Can be a Jinja2 template string (e.g. ``"{{question}}"``), a column name,
    or a callable. For ``loglikelihood`` tasks this is the context preceding
    the target."""

    doc_to_choice: str | Callable[[Doc], list[str]] | list[str] | None = None
    """Defines the set of answer choices for multiple-choice tasks.
    Can be a Jinja2 template (e.g. ``"{{choices.text}}"``), a column name,
    a static list of strings, or a callable returning a list of strings."""

    doc_to_target: str | Callable[[Doc], str | int | list[int] | list[str]] | None = (
        None
    )
    """The gold-standard target for each document.
    Can be the column name, a Jinja2 template, or a callable. For multiple-choice
    tasks this is typically the integer index into ``doc_to_choice`` (e.g. ``"{{answer}}"``).
    For generation tasks, it is the expected answer string."""

    gen_prefix: str | None = None
    """A string or Jinja2 template appended after the prompt (and choices, if any) but before
    the model generates or the target is scored. With a chat template, this
    is appended after the ``<|assistant|>`` token; without one it is appended
    to the end of the prompt. Useful for answer cues like ``"The answer is"``."""

    doc_to_image: Callable[[Doc], Any] | str | None = None
    """Extracts an image from the document for multimodal models.
    Can be a column name or a callable returning image data."""

    doc_to_audio: Callable[[Doc], Any] | str | None = None
    """Extracts audio from the document for multimodal models.
    Can be a column name or a callable returning audio data."""

    process_results: (
        Callable[
            [dict[str, Any], Sequence[LLOutput] | Sequence[Completion]],
            dict[str, list[Any]],
        ]
        | None
    ) = None
    """Custom post-processing of model outputs for metric computation.
    Receives ``(doc, results)`` and returns a dict mapping metric names to
    lists of values. Typically set in YAML via ``!function utils.xxx``."""

    target_delimiter: str = " "
    """String inserted between the input (prompt/choices) and the target
    output for each example (both few-shot and the test document)."""

    fewshot_delimiter: str = "\n\n"
    """String inserted between consecutive few-shot examples.
    Also used as the default ``until`` stop sequence for generation."""

    fewshot_config: dict[str, Any] | FewshotConfig | None = None
    """Advanced few-shot configuration. Accepts a dict or ``FewshotConfig``
    to override how few-shot examples are sampled and formatted (e.g.
    separate ``doc_to_text`` for examples, custom sampler, fixed indices).
    When None or a dict, it is converted to ``FewshotConfig`` in ``__post_init__``."""

    # ── Runtime configuration ───────────────────────────────────────────

    num_fewshot: int | None = None
    """Number of few-shot examples to prepend to each prompt. When None,
    the value is determined at runtime (typically by CLI ``--num_fewshot``)."""

    generation_kwargs: GenKwargs = field(default_factory=dict)
    """Keyword arguments for text generation (e.g. ``temperature``, ``until``,
    ``max_gen_toks``, ``do_sample``). Only relevant when ``output_type``
    is ``"generate_until"``. If empty, greedy defaults are applied."""

    # ── Scoring options ─────────────────────────────────────────────────

    metric_list: list[MetricConfig] = field(default_factory=list)
    """List of metrics to compute on model outputs. Each entry specifies
    a metric name, optional aggregation function, and whether higher is
    better (e.g. ``[{"metric": "exact_match", "higher_is_better": true}]``)."""

    filter_list: list[FilterPipeline] = field(default_factory=list)
    """List of named filter pipelines applied to model outputs before scoring.
    Each pipeline is a sequence of filter steps (e.g. regex extraction,
    stripping) and can carry its own ``metric_list``. Pipelines run
    independently on the same model outputs, allowing multiple scoring
    strategies from a single evaluation run (e.g. ``"strict-match"``
    and ``"maj@64"`` on GSM8k)."""

    scorer: str | ScorerConfig | None = None
    """A registered scorer name or inline scorer config. When set, scoring
    is delegated to this scorer instead of the default metric pipeline.

    Accepts a string (e.g. ``"first_token"``) which is normalised to
    ``{"type": "first_token"}`` in ``__post_init__``, or a full
    [ScorerConfig][ScorerConfig] dict with extra kwargs forwarded to the scorer
    constructor."""

    repeats: int = 1
    """Number of times to repeat each instance. Only used for generation
    tasks. Useful for sampling diversity (e.g. pass@k, self-consistency)."""

    unsafe_code: bool = False
    """Set to True to enable execution of untrusted code (e.g. for
    code-execution benchmarks). Must be explicitly opted in."""

    use_prompt: str | None = None
    """Name of a registered prompt template to apply (e.g.
    ``"promptsource:GPT-3 Style"``). When set, overrides ``doc_to_text``,
    ``doc_to_target``, and ``doc_to_choice``."""

    multiple_inputs: bool = False
    """Only for ``multiple_choice`` tasks. When True, ``doc_to_text`` returns
    a list of strings (one per choice) and ``doc_to_choice`` returns a 1 elememnt list.
    Each choice produces a different context scored via
    loglikelihood (e.g. Winogrande, where each option fills a blank)."""

    multiple_targets: bool = False
    """When True, ``doc_to_target`` may return a list of acceptable answers.
    Scoring considers any match a success."""

    should_decontaminate: bool = False
    """Whether to run decontamination checks against training data."""

    doc_to_decontamination_query: str | None = None
    """Jinja2 template or callable that extracts the decontamination query
    string from a document. Used when ``should_decontaminate`` is True.
    Falls back to ``doc_to_text`` if left as None."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata dict stored alongside results. Most tasks should include a
    ``version`` key. The ``num_fewshot`` key overrides the displayed n-shot
    column in result tables. Also passed to ``custom_dataset`` at runtime to pass arbitrary kwargs"""

    _formats_selection: str | None = None
    """Internal field. Holds the ``@suffix`` parsed from the task name
    (e.g. ``"cloze"`` from ``"arc_easy@cloze"``). Set automatically
    in ``__post_init__``; should not be set in YAML configs."""

    _qualified_name: str | None = None
    """Namespaced identity for pipeline tracking (e.g. ``"group_a::arc_easy"``).
    Set automatically by the factory when a task is built as a group member.
    Falls back to ``task`` when not set. Not intended for manual YAML use."""

    def __post_init__(self) -> None:
        self._resolve_formats()
        self._apply_generation_defaults()
        self._build_fewshot_config()
        self._normalize_scoring_config()

    def _resolve_formats(self) -> None:
        """Parse ``@formats`` from the task name and apply formats overrides."""
        # Extract @formats from the task name as selection (e.g. "arc_easy@cloze")
        # Runtime _formats_selection takes priority over the YAML task name.
        if self._formats_selection is None and "@" in self.task:
            self._formats_selection = self.task.rsplit("@", 1)[1]

        # If no formats: field, the selection itself becomes the formats
        if self.formats is None:
            self.formats = self._formats_selection

        # Resolve formats: it consumes doc_to_* as inputs (field mappings)
        # and returns overrides that are applied unconditionally.
        if self.formats is not None:
            from lm_eval.config.formats import FormatConfig

            resolved = FormatConfig.get(self.formats, selection=self._formats_selection)
            if resolved is not None:
                overrides = resolved.to_task_config(
                    doc_to_text=self.doc_to_text
                    if self.doc_to_text is not None
                    else "question",  # type: ignore
                    doc_to_choice=self.doc_to_choice,  # type: ignore
                    doc_to_target=self.doc_to_target
                    if self.doc_to_target is not None
                    else "answer",  # type: ignore
                )
                for key, value in overrides.items():
                    setattr(self, key, value)

    def _apply_generation_defaults(self) -> None:
        """Fill in missing ``generation_kwargs`` with sensible defaults."""
        if self.generation_kwargs:
            if self.output_type != "generate_until":
                eval_logger.warning(
                    "[%s] passed `generation_kwargs`, but not using `output_type: generate_until`!",
                    self.task,
                )

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(
                    self.generation_kwargs["temperature"]
                )

            if "until" not in self.generation_kwargs:
                eval_logger.warning(
                    "[%s]: No `until` specified in `generation_kwargs`! Defaulting to the fewshot_delimiter=%r",
                    self.task,
                    self.fewshot_delimiter,
                )
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if self.output_type == "generate_until":
                # ensure that we greedily generate in absence of explicit arguments otherwise
                self.generation_kwargs = cast(
                    "GenKwargs", default_gen_kwargs(self.fewshot_delimiter)
                )
                eval_logger.warning(
                    "[%s]: No `generation_kwargs` specified in task config, defaulting to %s",
                    self.task,
                    self.generation_kwargs,
                )

    def _build_fewshot_config(self) -> None:
        """Convert a raw ``fewshot_config`` dict to a [FewshotConfig][FewshotConfig], inheriting task-level fields."""
        if isinstance(self.fewshot_config, FewshotConfig):
            return
        if isinstance(self.fewshot_config, dict) or self.fewshot_config is None:
            self.fewshot_config = FewshotConfig.from_dict(
                self.fewshot_config or {},
                split=self.fewshot_split,
                process_docs=self.process_docs,
                fewshot_delimiter=self.fewshot_delimiter,
                target_delimiter=self.target_delimiter,
                gen_prefix=self.gen_prefix,
                doc_to_text=self.doc_to_text,
                doc_to_choice=self.doc_to_choice,
                doc_to_target=self.doc_to_target,
            )

    def _normalize_scoring_config(self) -> None:
        """Canonicalise scoring fields into a fully resolved ``filter_list``.

        After this method:

        * ``metric_list`` contains only what the user explicitly provided
          (empty if nothing was specified — registry defaults are the
          scorer layer's responsibility).
        * ``filter_list`` always has at least one entry.  Each entry
          carries a ``metric_list`` (per-pipeline if provided, otherwise
          the task-level ``metric_list`` as fallback).
        * ``scorer`` is normalised to [ScorerConfig][ScorerConfig] | None.
        """
        self.metric_list = normalize_metric_list(self.metric_list)
        self.filter_list = normalize_filter_list(self.filter_list)

        # Distribute task-level metrics as fallback into pipelines that
        # don't have their own.
        for pipeline in self.filter_list:
            pipeline.setdefault("metric_list", list(self.metric_list))

        # When no filter_list is provided, create a default scorer entry.
        # ``filter`` is empty so that the scorer class can apply its own
        # ``default_filter_cfg`` (the ``or`` chain in ``_build_filter``
        # treats ``[]`` the same as a missing key).
        if not self.filter_list:
            self.filter_list = [
                {"name": "none", "filter": [], "metric_list": list(self.metric_list)}
            ]

        # Normalise the scorer type to a dict form for consistent downstream handling.
        if isinstance(self.scorer, str):
            self.scorer = {"type": self.scorer}

    def to_dict(self, keep_callable: bool = False) -> dict[str, str]:
        """Dumps the current config as a dictionary object, as a printable format.

        null fields will not be printed.
        Used for dumping results alongside a full task configuration

        :return: dict
            A printable dictionary version of the TaskConfig object.

        """
        from lm_eval.config.utils import serialize_config

        return serialize_config(self, keep_callable=keep_callable)
