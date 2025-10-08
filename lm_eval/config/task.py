from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Union

import datasets

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import OutputType
from lm_eval.config.metric import MetricConfig
from lm_eval.config.utils import maybe_serialize


if TYPE_CHECKING:
    from lm_eval.api.samplers import ContextSampler
    from lm_eval.api.task import Task
    from lm_eval.config.template import TemplateConfig

eval_logger = logging.getLogger(__name__)

DataSet = Union[datasets.Dataset, Iterable[dict[str, Any]]]
DSplits = dict[str, DataSet]


@dataclass
class RepeatConfig:
    """Encapsulates information about a single repeat."""

    repeats: int = 1
    metric_fn: str | Callable = "pass@N"
    kwargs: dict | None = field(default_factory=dict)


@dataclass
class FilterConfig:
    """Encapsulates information about a single filter pipeline."""

    name: str
    ensemble: FilterEnsemble
    metric_list: list[MetricConfig]


@dataclass
class FewshotConfig:
    # hack: this returns task.config.num_fewshot
    # to keep in sync as it is runtime-modified
    num_fewshot: Callable[[], int]
    split: str | None = None
    sampler: str | Callable = "default"
    samples: Callable[[], DataSet] | DataSet | None = None
    process_docs: Callable[[DataSet], DataSet] | None = None
    fewshot_indices: list[int] | None = None
    rnd: int = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.samples is not None and not (
            isinstance(self.samples, list) or callable(self.samples)
        ):
            raise TypeError(
                "samples must be either list[dict] or callable returning list[dict]"
            )

        if self.split is not None and self.samples is not None:
            eval_logger.warning(
                "Both split and samples are configured; split will take precedence"
            )

    @property
    def has_source(self) -> bool:
        """Check if any fewshot source is configured."""
        return self.split is not None or self.samples is not None

    def _get_raw_docs(self, dataset: DSplits) -> DataSet | None:
        """Get raw documents from configured source."""
        if self.split is not None:
            return dataset[self.split]

        if self.samples is not None:
            if isinstance(self.samples, list):
                return self.samples
            elif callable(self.samples):
                # If samples is a callable, it should return a list of dicts
                return self.samples()
            else:
                raise TypeError(
                    "samples must be either a list of dicts or a callable returning a list"
                )

    def get_docs(self, dataset) -> DataSet | None:
        """Get processed documents from configured source."""
        raw_docs = self._get_raw_docs(dataset)
        if raw_docs is None:
            return None

        if self.process_docs is not None:
            return self.process_docs(raw_docs)
        return raw_docs

    @property
    def get_sampler(self) -> Callable[..., Any] | None:
        from lm_eval.api import samplers

        if isinstance(self.sampler, str):
            return samplers.get_sampler(self.sampler)
        elif callable(self.sampler):
            return self.sampler

    def init_sampler(
        self, docs: list[dict], task: Task, rnd=None, fewshot_indices=None
    ) -> ContextSampler:
        """Initialize the sampler with the given documents and task."""
        if rnd is None:
            raise ValueError(
                "A `random.Random` generator argument must be provided to `rnd` of FewShotSampler!"
            ) from None
        return self.get_sampler(
            docs,
            rnd=rnd,
            fewshot_indices=fewshot_indices
            if fewshot_indices
            else self.fewshot_indices,
        )  # type: ignore


@dataclass
class TaskConfig:
    # task naming/registry
    task: str | None = None
    task_alias: str | None = None
    tag: str | list | None = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    custom_dataset: Callable[..., DataSet] | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_kwargs: dict | None = field(default_factory=dict)
    training_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    fewshot_split: str | None = None
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Callable[[DataSet], DataSet] | None = None
    doc_to_text: Callable[[dict[str, Any]], Any] | str | None = None
    doc_to_target: Callable[[dict[str, Any]], Any] | str | None = None
    doc_to_image: Callable[[dict[str, Any]], Any] | str | None = None
    doc_to_audio: Callable[[dict[str, Any]], Any] | str | None = None
    unsafe_code: bool = False
    doc_to_choice: Callable[[dict[str, Any]], Any] | str | dict | list | None = None
    process_results: (
        Callable[[dict[str, Any], list[Any]], dict[str, Any]] | str | None
    ) = None
    use_prompt: str | None = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: dict[str, Any] | None = None
    # runtime configuration options
    num_fewshot: int | None = None
    generation_kwargs: dict[str, Any] | None = None
    # scoring options
    metric_list: list | None = None
    output_type: OutputType = "generate_until"
    repeats: int = 1
    filter_list: list[dict] | None = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: str | None = None
    gen_prefix: str | None = None
    multiple_inputs: bool = False
    multiple_targets: bool = False
    metadata: dict = field(
        default_factory=dict
    )  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    _metric_list: list[MetricConfig] = field(default_factory=list)
    _filter_list: list[FilterConfig] = field(default_factory=list)
    # ds_cfg: DatasetConfig = field(init=False)
    fewshot_cfg: FewshotConfig = field(init=False)
    _fn: dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ### ---setup generation kwargs--- ###
        if self.generation_kwargs is not None:
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
                self.generation_kwargs = {
                    "until": (
                        None
                        if self.fewshot_delimiter is None
                        else [self.fewshot_delimiter]
                    ),
                    "do_sample": False,
                    "temperature": 0,
                }
                eval_logger.warning(
                    f"{self.task}: No `generation_kwargs` specified in task config, defaulting to {self.generation_kwargs}"
                )
        # ---setup fewshot config--- #
        _fewshot_cfg = self.fewshot_config if self.fewshot_config is not None else {}
        self.fewshot_cfg = FewshotConfig(
            num_fewshot=lambda: self.num_fewshot or _fewshot_cfg.get("num_fewshot", 0),
            split=self.fewshot_split,
            sampler=_fewshot_cfg.get("sampler", "default"),
            samples=_fewshot_cfg.get("samples", None),
            process_docs=_fewshot_cfg.get("process_docs", None),
            fewshot_indices=_fewshot_cfg.get("fewshot_indices", None),
        )

    def _get_metric(self, metric_list: list[dict] | None = None) -> list[MetricConfig]:
        from lm_eval.api.registry import (
            AGGREGATION_REGISTRY,
            DEFAULT_METRIC_REGISTRY,
            get_aggregation,
            get_metric,
            get_metric_aggregation,
            is_higher_better,
        )

        # if metric_list defined inside a filter, use that; otherwise use the task's metric_list
        metric_list = metric_list or self.metric_list
        metrics = []
        if not metric_list:
            # ---------- 1. If no metrics defined, use defaults for output type ----------
            _metric_list = DEFAULT_METRIC_REGISTRY[self.output_type]
            eval_logger.info(
                f"No metrics defined in config, using default metrics for {self.output_type}={_metric_list}"
            )
            metrics.extend(
                MetricConfig(
                    name=metric_name,
                    fn=get_metric(metric_name),
                    aggregation_fn=get_metric_aggregation(metric_name),
                    higher_is_better=is_higher_better(metric_name) or True,
                )
                for metric_name in _metric_list
            )
        else:
            # ---------- 2. Process user-defined metrics from config ----------
            for metric_config in metric_list:
                metric_name = metric_config["metric"]
                _metric_fn_kwargs = {
                    key: metric_config[key]
                    for key in metric_config
                    if key
                    not in ["metric", "aggregation", "higher_is_better", "hf_evaluate"]
                }
                _hf_evaluate_metric: bool = metric_config.get("hf_evaluate", False)
                _metric_fn = None
                _aggregation = None

                if self.process_results is not None:
                    # User will compute metrics inside `process_results()`
                    _metric_name = None
                    _metric_fn_kwargs = {}
                elif callable(metric_name):
                    # User passed a function object
                    _metric_name = metric_name.__name__
                    _metric_fn = metric_name.__call__
                else:
                    # Normal: look up by name
                    _metric_name = metric_name
                    _metric_fn = get_metric(metric_name, _hf_evaluate_metric)

                # ---------- 3. Decide how to aggregate examples ----------
                if "aggregation" in metric_config:
                    if isinstance(_agg_name := metric_config["aggregation"], str):
                        _aggregation = get_aggregation(_agg_name)
                    elif callable(_agg_name):  # noqa: E721
                        _aggregation = metric_config["aggregation"]
                else:
                    INV_AGG_REGISTRY = {v: k for k, v in AGGREGATION_REGISTRY.items()}
                    _aggregation = get_metric_aggregation(metric_name)
                    eval_logger.warning(
                        f"[Task: {self.task}] metric {metric_name} is defined, but aggregation is not. "
                        f"using default "
                        f"aggregation={INV_AGG_REGISTRY[_aggregation]}"
                    )

                # ---------- 4. Determine “higher-is-better” semantics ----------
                if "higher_is_better" in metric_config:
                    _higher_is_better = metric_config["higher_is_better"]
                else:
                    eval_logger.warning(
                        f"[Task: {self.task}] metric {metric_name} is defined, but higher_is_better is not. "
                        f"using default "
                        f"higher_is_better={is_higher_better(metric_name)}"
                    )
                    _higher_is_better = is_higher_better(metric_name)

                metrics.append(
                    MetricConfig(
                        name=_metric_name,
                        fn=_metric_fn,
                        kwargs=_metric_fn_kwargs,
                        aggregation_fn=_aggregation,
                        higher_is_better=_higher_is_better,
                        hf_evaluate=_hf_evaluate_metric,
                    )
                )
        for m in metrics:
            if m not in self._metric_list:
                self._metric_list.append(m)
        return metrics

    @property
    def get_filters(self) -> list[FilterConfig]:
        from lm_eval.filters import build_filter_ensemble

        if not self.filter_list:
            eval_logger.debug(
                "No custom filters defined; falling back to 'take_first' for handling repeats."
            )
            return [
                FilterConfig(
                    name="none",
                    ensemble=build_filter_ensemble("none", [("take_first", None)]),
                    metric_list=self._get_metric(metric_list=None),
                )
            ]
        else:

            def _strip_fn(d: dict) -> tuple[str, dict]:
                return d["function"], {
                    k: v for k, v in d.items() if k not in ["function", "metric_list"]
                }

            configs = (
                self.filter_list.values()
                if isinstance(self.filter_list, dict)
                else self.filter_list
            )
            x = [
                FilterConfig(
                    name=cfg["name"],
                    ensemble=build_filter_ensemble(
                        filter_name=cfg["name"],
                        components=[_strip_fn(f) for f in cfg["filter"]],
                    ),
                    metric_list=self._get_metric(metric_list=cfg.get("metric_list")),
                )
                for cfg in configs
            ]
            return x

    @classmethod
    def from_yaml(cls, data: dict[str, Any]) -> TaskConfig:
        """Create a TaskConfig instance from a YAML-like dictionary."""
        fn = {k: v for k, v in data.items() if callable(v)}
        return cls(**data, _fn=fn)

    @classmethod
    def from_template(cls, template: TemplateConfig, **kwargs) -> TaskConfig:
        """Create a TaskConfig instance from a template.

        Args:
            template: TemplateConfig instance (MCQTemplateConfig or ClozeTemplateConfig)
            **kwargs: Additional arguments to override template defaults

        Returns:
            TaskConfig instance configured from the template
        """
        from lm_eval.config.template import (
            ClozeTemplateConfig,
            MCQTemplateConfig,
        )

        # Extract base configuration from template
        config_dict = {
            "task": template.task,
            "doc_to_text": template.doc_to_text,
            "doc_to_choice": template.doc_to_choice,
            "doc_to_target": template.doc_to_target,
            "description": template.description,
            "target_delimiter": template.target_delimiter,
            "fewshot_delimiter": template.fewshot_delimiter,
            "metric_list": template.metric_list,
        }

        # Add common template attributes if they exist
        if hasattr(template, "answer_suffix"):
            config_dict["target_delimiter"] = (
                template.answer_suffix + template.target_delimiter
            )

        # Handle template-specific configurations
        if isinstance(template, MCQTemplateConfig):
            # For MCQ templates, set up multiple choice specific config
            config_dict["output_type"] = "multiple_choice"

            # MCQ templates typically use accuracy metrics
            if template.metric_list is None:
                config_dict["metric_list"] = [{"metric": "acc"}]

        elif isinstance(template, ClozeTemplateConfig):
            # For Cloze templates, set up generation config
            config_dict["output_type"] = "generate_until"

            # Cloze templates typically use accuracy and normalized accuracy
            if template.metric_list is None:
                config_dict["metric_list"] = [{"metric": "acc"}, {"metric": "acc_norm"}]
        else:
            # Generic template - try to infer output type
            if hasattr(template, "template"):
                if template.template == "mcq":
                    config_dict["output_type"] = "multiple_choice"
                elif template.template == "cloze":
                    config_dict["output_type"] = "generate_until"

        # Override with any user-provided kwargs
        config_dict.update(kwargs)

        # Create and return TaskConfig instance
        return cls(**config_dict)

    def to_dict(self, keep_callable: bool = False) -> dict:
        def _ser(x):
            if isinstance(x, dict):
                return {k: _ser(v) for k, v in x.items()}
            if isinstance(x, (list, tuple, set)):
                return type(x)(_ser(i) for i in x)
            return maybe_serialize(x, keep_callable)

        return {k: _ser(v) for k, v in asdict(self).items() if v is not None}
