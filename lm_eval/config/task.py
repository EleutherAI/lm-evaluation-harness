import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union

from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.instance import OutputType
from lm_eval.config.metric import MetricConfig
from lm_eval.config.utils import maybe_serialize


if TYPE_CHECKING:
    from lm_eval.api.samplers import ContextSampler
    from lm_eval.api.task import Task, eval_logger

eval_logger = logging.getLogger(__name__)


@dataclass
class RepeatConfig:
    """Encapsulates information about a single repeat."""

    repeats: int = 1
    metric_fn: Union[str, Callable] = "pass@N"
    kwargs: Optional[dict] = None


@dataclass
class FilterConfig:
    """Encapsulates information about a single filter."""

    name: str
    fn: Optional[Callable] = None
    kwargs: Optional[dict] = None


@dataclass
class FewshotConfig:
    num: int = 0
    split: Optional[str] = None
    sampler: Union[str, Callable] = "default"
    samples: Union[Callable[[], list[dict]], list[dict], None] = None
    process_docs: Optional[Callable[[list[dict]], Iterable[dict]]] = None
    fewshot_indices: Optional[list[int]] = None
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

    def _get_raw_docs(
        self, dataset
    ) -> Union[list[dict], Callable[[], Iterable[dict]], None]:
        """Get raw documents from configured source."""
        if self.split is not None:
            return dataset[self.split]

        if self.samples is not None:
            if isinstance(self.samples, list):
                return self.samples
            elif callable(self.samples):
                return self.samples
            else:
                raise TypeError(
                    "samples must be either a list of dicts or a callable returning a list"
                )

    def get_docs(self, dataset) -> Optional[Iterable[dict]]:
        """Get processed documents from configured source."""
        raw_docs = self._get_raw_docs(dataset)
        if raw_docs is None:
            return None

        if self.process_docs is not None:
            return self.process_docs(raw_docs)
        return raw_docs

    @property
    def get_sampler(self):
        from lm_eval.api import samplers

        if isinstance(self.sampler, str):
            return samplers.get_sampler(self.sampler)
        elif callable(self.sampler):
            return self.sampler

    def init_sampler(
        self, docs: list[dict], task: "Task", rnd=None, fewshot_indices=None
    ) -> "ContextSampler":
        """Initialize the sampler with the given documents and task."""
        if rnd is None:
            raise ValueError(
                "A `random.Random` generator argument must be provided to `rnd` of FewShotSampler!"
            )
        return self.get_sampler(
            docs,
            task,
            rnd=rnd,
            fewshot_indices=fewshot_indices
            if fewshot_indices
            else self.fewshot_indices,
        )


@dataclass
class DatasetConfig:
    """Encapsulates information about a dataset."""

    path: Optional[str] = None
    name: Optional[str] = None
    kwargs: Optional[dict] = field(default_factory=dict)
    custom: Optional[Callable] = None
    metadata: Optional[dict] = None


@dataclass
class TaskConfig(dict):
    # task naming/registry
    task: Optional[str] = None
    task_alias: Optional[str] = None
    tag: Optional[Union[str, list]] = None
    # HF dataset options.
    # which dataset to use,
    # and what splits for what purpose
    custom_dataset: Optional[Callable] = None
    dataset_path: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_kwargs: Optional[dict] = None
    training_split: Optional[str] = None
    validation_split: Optional[str] = None
    test_split: Optional[str] = None
    fewshot_split: Optional[str] = (
        None  # TODO: assert that this not None if num_fewshot > 0. (?) assert if this is same split as one evaluating (?)
    )
    # formatting / prompting options.
    # see docs/advanced_task_guide.md for more info
    process_docs: Optional[Callable] = None
    doc_to_text: Optional[Union[Callable, str]] = None
    doc_to_target: Optional[Union[Callable, str]] = None
    doc_to_image: Union[Callable, str, None] = None
    doc_to_audio: Union[Callable, str, None] = None
    unsafe_code: bool = False
    doc_to_choice: Optional[Union[Callable, str, dict, list]] = None
    process_results: Optional[Union[Callable, str]] = None
    use_prompt: Optional[str] = None
    description: str = ""
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"
    fewshot_config: Optional[dict] = None
    # runtime configuration options
    num_fewshot: Optional[int] = 0
    # scoring options
    metric_list: Optional[list] = None
    output_type: OutputType = "generate_until"
    generation_kwargs: Optional[dict] = None
    repeats: int = 1
    filter_list: Optional[list[dict]] = None
    should_decontaminate: bool = False
    doc_to_decontamination_query: Optional[str] = None
    gen_prefix: Optional[str] = None
    metadata: Optional[dict] = (
        None  # by default, not used in the code. allows for users to pass arbitrary info to tasks
    )
    _metric_list: list[MetricConfig] = None
    _filter_list: list[FilterConfig] = None
    ds_cfg: DatasetConfig = None
    fewshot_cfg: FewshotConfig = None

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
        # ---setup dataset config--- #
        self.ds_cfg = DatasetConfig(
            path=self.dataset_path,
            name=self.dataset_name,
            kwargs=self.dataset_kwargs,
            custom=self.custom_dataset,
            metadata=self.metadata,
        )
        # ---setup fewshot config--- #
        _fewshot_cfg = self.fewshot_config if self.fewshot_config is not None else {}
        self.fewshot_cfg = FewshotConfig(
            split=self.fewshot_split,
            sampler=_fewshot_cfg.get("sampler", "default"),
            samples=_fewshot_cfg.get("samples", None),
            process_docs=_fewshot_cfg.get("process_docs", None),
            fewshot_indices=_fewshot_cfg.get("fewshot_indices", None),
        )

    @property
    def get_metrics(self) -> list["MetricConfig"]:
        from lm_eval.api.registry import (
            AGGREGATION_REGISTRY,
            DEFAULT_METRIC_REGISTRY,
            get_aggregation,
            get_metric,
            get_metric_aggregation,
            is_higher_better,
        )

        metrics = []
        if self.metric_list is None:
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
                    higher_is_better=is_higher_better(metric_name),
                )
                for metric_name in _metric_list
            )
        else:
            # ---------- 2. How will the outputs be evaluated ----------
            for metric_config in self.metric_list:
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
        return metrics

    @property
    def get_filters(self) -> list["FilterEnsemble"]:
        from lm_eval.filters import build_filter_ensemble

        if not self.filter_list:
            eval_logger.debug(
                "No custom filters defined; falling back to 'take_first' for handling repeats."
            )
            return [build_filter_ensemble("none", [["take_first", None]])]
        else:

            def _strip_fn(d: dict) -> dict:
                return {k: v for k, v in d.items() if k != "function"}

            configs = (
                self.filter_list.values()
                if isinstance(self.filter_list, dict)
                else self.filter_list
            )

            return [
                build_filter_ensemble(
                    filter_name=cfg["name"],
                    components=[[_strip_fn(f) for f in cfg["filter"]]],
                )
                for cfg in configs
            ]

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        return setattr(self, item, value)

    def to_dict(self, keep_callable: bool = False) -> dict:
        """Return a printable dict with Nones stripped and callables serialised.

        :return: dict
            A printable dictionary version of the TaskConfig object.
        """

        cfg = asdict(self)
        return {
            k: [
                {mk: maybe_serialize(mv, keep_callable) for mk, mv in md.items()}
                for md in v
            ]
            if k == "metric_list"
            else maybe_serialize(v)
            for k, v in cfg.items()
            if v is not None
        }
