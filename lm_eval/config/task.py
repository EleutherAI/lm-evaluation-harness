import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import Instance, OutputType
from lm_eval.api.registry import metric_agg_registry, metric_registry
from lm_eval.config.metric import MetricConfig
from lm_eval.config.template import Template, init_template
from lm_eval.config.utils import maybe_serialize
from lm_eval.types import DatasetSplits, TaskDataSet


if TYPE_CHECKING:
    from lm_eval.api.samplers import ContextSampler

eval_logger = logging.getLogger(__name__)


ALL_OUTPUT_TYPES = [
    "loglikelihood",
    "multiple_choice",
    "loglikelihood_rolling",
    "generate_until",
]


@dataclass
class RepeatConfig:
    """Encapsulates information about a single repeat."""

    repeats: int = 1
    metric_name: str = ""
    reducer: str | Callable[..., float] | None = "pass@k"
    kwargs: dict | None = field(default_factory=dict)

    def __post_init__(self):
        if self.repeats < 1:
            self.reducer = None
        if isinstance(self.reducer, str):
            self.metric_name = self.reducer
            self.reducer = metric_agg_registry.get(self.reducer)
        elif callable(self.reducer):
            self.reducer = self.reducer
            self.metric_name = self.reducer.__name__

    @classmethod
    def from_cfg(cls, cfg: dict | str | int):
        if not isinstance(cfg, dict):
            return cls(repeats=int(cfg), reducer="mean", metric_name="mean")
        return cls(**cfg)


@dataclass
class FilterConfig:
    """Encapsulates information about a single filter pipeline."""

    name: str
    ensemble: FilterEnsemble
    metric_list: list[MetricConfig]

    def _compute_metrics(self, instances: dict[str, list[Instance]]) -> dict[str, Any]:
        """Compute metrics for a single filter pipeline."""
        res = {}
        for metric in self.metric_list:
            for _, instance in instances.items():
                res[metric.name] = metric.fn(
                    (
                        instance[0].doc,
                        [instance.filtered_resps[self.name] for instance in instance],
                    ),
                )
                # res[f"{self.name}_{metric.name}"] = metric.fn(
                #     instance=instances[0],
                #     filtered_resps={
                #         instance.name: instance.filtered_resps[self.name]
                #         for instance in instances
                #     },
                #     **metric.kwargs,
                # )
            #
            # [res[metric.name] = metric.fn(
            #     instance[0].doc, [instance.filtered_resps[self.name] for instance in instance]
            # ) for instance in instances.values()]
        return res


@dataclass
class FewshotConfig:
    # hack: this returns task.config.num_fewshot
    # to keep in sync as it is runtime-modified
    num_fewshot: Callable[[], int]
    split: str | None = None
    sampler: str | Callable = "default"
    samples: Callable[[], TaskDataSet] | TaskDataSet | None = None
    process_docs: Callable[[TaskDataSet], TaskDataSet] | None = None
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

    def _get_raw_docs(self, dataset: DatasetSplits) -> TaskDataSet | None:
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

    def get_docs(self, dataset) -> TaskDataSet | None:
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
        self, docs: list[dict] | None = None, rnd=None, fewshot_indices=None
    ) -> "ContextSampler":
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


@dataclass(kw_only=True)
class TaskConfig:
    # task naming/registry
    task: str | None = None
    task_alias: str | None = None
    tag: str | list | None = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    custom_dataset: Callable[..., DatasetSplits] | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    dataset_kwargs: dict | None = field(default_factory=dict)
    training_split: str | None = None
    validation_split: str | None = None
    test_split: str | None = None
    fewshot_split: str | None = None
    template: Template | None = None
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Callable[[TaskDataSet], TaskDataSet] | None = None
    doc_to_text: Callable[[dict[str, Any]], str] | str | None = None
    doc_to_target: Callable[[dict[str, Any]], str | int] | str | int | None = None
    doc_to_image: Callable[[dict[str, Any]], Any] | str | None = None
    doc_to_audio: Callable[[dict[str, Any]], Any] | str | None = None
    unsafe_code: bool = False
    doc_to_choice: Callable[[dict[str, Any]], list[str]] | str | dict | list | None = (
        None
    )
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
    filter_list: list[dict] | None = None
    metric_list: list | None = None
    output_type: OutputType = "generate_until"
    repeats: int = 1
    should_decontaminate: bool = False
    doc_to_decontamination_query: str | None = None
    gen_prefix: str | None = None
    multiple_inputs: bool = False
    multiple_targets: bool = False
    unconditional_context: str = ""
    metadata: dict = field(
        default_factory=dict
    )  # by default, not used in the code. allows for users to pass arbitrary info to tasks

    _metric_list: list[MetricConfig] = field(default_factory=list)
    _filter_list: list[FilterConfig] = field(default_factory=list)
    # ds_cfg: DatasetConfig = field(init=False)
    _fewshot_cfg: FewshotConfig = field(init=False)
    repeat_cfg: RepeatConfig = field(
        default_factory=lambda: RepeatConfig.from_cfg(1), init=False
    )
    _fn: dict[str, Callable] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.output_type is not None and self.output_type not in ALL_OUTPUT_TYPES:
            raise ValueError(
                f"Got invalid output_type '{self.output_type}', must be in '{','.join(ALL_OUTPUT_TYPES)}'"
            )

        self._verify_gen_kwargs()

        # ---setup fewshot config--- #
        _fewshot_cfg = self.fewshot_config if self.fewshot_config is not None else {}
        self._fewshot_cfg = FewshotConfig(
            num_fewshot=lambda: self.num_fewshot or _fewshot_cfg.get("num_fewshot", 0),
            split=self.fewshot_split,
            sampler=_fewshot_cfg.get("sampler", "default"),
            samples=_fewshot_cfg.get("samples", None),
            process_docs=_fewshot_cfg.get("process_docs", None),
            fewshot_indices=_fewshot_cfg.get("fewshot_indices", None),
        )
        self.template = init_template(self.template)
        if self.template:
            self.doc_to_text = self.template.question or self.doc_to_text
            self.doc_to_target = self.template.target or self.doc_to_target
            self.doc_to_choice = self.template.choices or self.doc_to_choice

        if isinstance(self.repeats, str) and self.repeats.isdigit():
            self.repeats = int(self.repeats)
        # if isinstance(self.repeats, int):
        #     self.repeats = {"repeats": self.repeats}
        # assert isinstance(self.repeats, dict | RepeatConfig), (
        #     "repeats must be int or dict"
        # )

        self.repeat_cfg = RepeatConfig.from_cfg(self.repeats)

    def _get_metric(self, metric_list: list[dict] | None = None) -> list[MetricConfig]:
        from lm_eval.api.registry import (
            DEFAULT_METRIC_REGISTRY,
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
                metric_registry.get(metric_name) for metric_name in _metric_list
            )
        else:
            # ---------- 2. Process user-defined metrics from config ----------
            for metric_config in metric_list:
                metrics.append(
                    MetricConfig.from_yaml_field(metric_config, task=self.task or "")
                )
        for m in metrics:
            if m not in self._metric_list:
                self._metric_list.append(m)
        return metrics

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

    def _verify_gen_kwargs(self) -> None:
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
                if isinstance(self.generation_kwargs["until"], str):
                    self.generation_kwargs["until"] = [self.generation_kwargs["until"]]
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

    @classmethod
    def from_yaml(cls, data: dict[str, Any]):
        """Create a TaskConfig instance from a YAML-like dictionary."""
        fn = {k: v for k, v in data.items() if callable(v)}
        return cls(**data, _fn=fn)

    @classmethod
    def from_arbitrary_dict(cls, data: dict[str, Any]):
        """Create a TaskConfig instance from a dictionary."""
        _fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in _fields})

    def to_dict(self, keep_callable: bool = False) -> dict:
        def _ser(x):
            if isinstance(x, dict):
                return {k: _ser(v) for k, v in x.items()}
            if isinstance(x, (list, tuple, set)):
                return type(x)(_ser(i) for i in x)
            return maybe_serialize(x, keep_callable)

        return {k: _ser(v) for k, v in asdict(self).items() if v is not None}
