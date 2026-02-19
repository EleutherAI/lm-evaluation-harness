from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from inspect import getsource
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from typing_extensions import TypedDict

from lm_eval.defaults import default_gen_kwargs


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    import datasets

    from lm_eval.api.instance import OutputType

Doc = dict[str, Any]
# A single dataset split – iterable + sized collection of docs.
# datasets.Dataset is the primary impl; list[Doc] works for custom datasets.
DataSplit: TypeAlias = "datasets.Dataset | Sequence[Doc]"

# The full dataset – maps split names (training_split, test_split, etc.) to splits.
# datasets.DatasetDict is the primary impl; dict[str, DataSplit] works too.
Dataset: TypeAlias = "Mapping[str, DataSplit]"


eval_logger = logging.getLogger(__name__)


class _MetricConfig(TypedDict, total=False):
    metric: str | Callable
    aggregation: str | Callable | None
    reduction: str | Callable | None
    higher_is_better: bool | None
    kwargs: dict[str, Any] | None


class FilterConfig(TypedDict, total=False):
    function: str
    kwargs: dict[str, str]
    metric_list: list[_MetricConfig] | None


class FilterList(TypedDict, total=False):
    name: str
    filters: list[FilterConfig]


@dataclass
class FewshotConfig:
    """Configuration for few-shot example formatting.

    These fields override the parent TaskConfig fields when formatting
    few-shot examples (as opposed to the test example).

    note: num_fewshot is also runtime dependent, so is not included here.
    """

    sampler: str = "default"
    split: str | None = None
    process_docs: Callable[..., list[dict[str, Any]]] | None = None
    fewshot_indices: list[int] | None = None
    samples: list[dict[str, Any]] | Callable[[], list[dict[str, Any]]] | None = None
    # Override doc formatting for fewshot examples
    doc_to_text: str | Callable[[Doc], str] | None = None
    doc_to_choice: str | Callable[[Doc], list[str]] | list[str] | None = None
    doc_to_target: str | Callable[[Doc], str | int] | None = None
    gen_prefix: str | None = None
    fewshot_delimiter: str | None = None
    target_delimiter: str | None = None

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
                case _: raise Exception(
                        "`fewshot_config['samples']` was incorrectly defined in the configuration. It should either be `list[dict]`, or callable returning this list."
                    ) from None
            # fmt: on

    @classmethod
    def from_dict(
        cls,
        cfg: dict,
        *,
        # inherited from TaskConfig if not specified
        fewshot_split: str | None = None,
        process_docs: Callable[[Iterable[dict[str, Any]]], list[dict[str, Any]]]
        | None = None,
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
    # task naming/registry
    task: str
    task_alias: str | None = None
    output_type: OutputType = "generate_until"
    tag: list[str] = field(default_factory=list)
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    custom_dataset: Callable[..., Dataset] | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_kwargs: dict[str, str | int | float] = field(default_factory=dict)
    training_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    fewshot_split: str | None = (
        None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaluating (?)
    )
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Callable[..., list[dict[str, Any]]] | None = None
    description: str = ""
    doc_to_text: Callable[[Doc], str | list[str]] | str | None = None
    doc_to_choice: Callable[[Doc], list[str]] | str | list[str] | None = None
    doc_to_target: Callable[[Doc], str | int | list[int] | list[str]] | str | None = (
        None
    )
    gen_prefix: str | None = None
    doc_to_image: Callable[[Doc], Any] | str | None = None
    doc_to_audio: Callable[[Doc], Any] | str | None = None
    process_results: (
        Callable[[dict[str, Any], list[str]], dict[str, list[Any]]] | str | None
    ) = None
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict[str, Any] | FewshotConfig | None = None
    # runtime configuration options
    num_fewshot: int | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    # scoring options
    metric_list: list[_MetricConfig] = field(default_factory=list)
    filter_list: FilterList = field(default_factory=list)
    scorer: str | None = None
    repeats: int = 1
    unsafe_code: bool = False
    use_prompt: str | None = None
    multiple_inputs: bool = False
    multiple_targets: bool = False
    should_decontaminate: bool = False
    doc_to_decontamination_query: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.generation_kwargs:
            if self.output_type != "generate_until":
                eval_logger.warning(
                    f"[{self.task}] passed `generation_kwargs`, but not using `output_type: generate_until`!"
                )

            if "temperature" in self.generation_kwargs:
                self.generation_kwargs["temperature"] = float(
                    self.generation_kwargs["temperature"]
                )

            if "until" not in self.generation_kwargs:
                eval_logger.warning(
                    f"{self.task}: No `until` specified in `generation_kwargs`! Defaulting to the fewshot_delimiter={repr(self.fewshot_delimiter)}"
                )
                self.generation_kwargs["until"] = [self.fewshot_delimiter]
        else:
            if self.output_type == "generate_until":
                # ensure that we greedily generate in absence of explicit arguments otherwise
                self.generation_kwargs = default_gen_kwargs(self.fewshot_delimiter)
                eval_logger.warning(
                    f"{self.task}: No `generation_kwargs` specified in task config, defaulting to {self.generation_kwargs}"
                )
        self.fewshot_config = (
            FewshotConfig.from_dict(
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
            if (isinstance(self.fewshot_config, dict) or self.fewshot_config is None)
            else self.fewshot_config
        )

    def to_dict(self, keep_callable: bool = False) -> dict:
        """Dumps the current config as a dictionary object, as a printable format.

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
            elif k == "metric_list":
                for metric_dict in v:
                    for metric_key, metric_value in metric_dict.items():
                        if callable(metric_value):
                            metric_dict[metric_key] = self.serialize_function(
                                metric_value, keep_callable=keep_callable
                            )
                cfg_dict[k] = v
            elif callable(v):
                cfg_dict[k] = self.serialize_function(v, keep_callable=keep_callable)
        return cfg_dict

    def serialize_function(
        self, value: Callable | str, keep_callable=False
    ) -> Callable | str:
        """Serializes a given function or string.

        If 'keep_callable' is True, the original callable is returned.
        Otherwise, attempts to return the source code of the callable using 'getsource'.
        """
        if keep_callable:
            return value
        else:
            try:
                return getsource(value)  # type:ignore[invalid-argument-type]
            except (TypeError, OSError):
                return str(value)
