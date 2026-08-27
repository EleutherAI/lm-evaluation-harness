import copy
import logging
from typing import Any

from lm_eval.loggers.utils import (  # noqa: F401
    _handle_non_serializable,
    remove_none_pattern,
)


logger = logging.getLogger(__name__)


def _sample_to_trace(
    trackio_module, sample: dict[str, Any], task_name: str, output_type: str
):
    """Build a trackio.Trace from a single eval sample.

    Picks the most natural prompt/response pair for each task output_type and
    attaches the gold target plus any computed metrics as metadata so the
    Trackio dashboard surfaces everything alongside the conversation.
    """
    prompt = ""
    response = ""

    args = sample.get("arguments") or {}
    if output_type == "loglikelihood":
        first = (
            args.get("arg_0", {})
            if isinstance(args, dict)
            else (args[0] if args else {})
        )
        prompt = (
            first.get("arg_0", "")
            if isinstance(first, dict)
            else (first[0] if first else "")
        )
        filt = sample.get("filtered_resps")
        if filt:
            logprob, is_greedy = (
                filt[0] if isinstance(filt[0], (list, tuple)) else (filt[0], None)
            )
            response = f"continuation log-prob: {logprob}" + (
                f"\n\ngreedy={bool(is_greedy)}" if is_greedy is not None else ""
            )
    elif output_type == "multiple_choice":
        first = (
            args.get("arg_0", {})
            if isinstance(args, dict)
            else (args[0] if args else {})
        )
        prompt = (
            first.get("arg_0", "")
            if isinstance(first, dict)
            else (first[0] if first else "")
        )
        filt = sample.get("filtered_resps") or []
        try:
            import numpy as np

            choice_idx = int(
                np.argmax([n[0] if isinstance(n, (list, tuple)) else n for n in filt])
            )
            response = f"selected choice {choice_idx}"
        except Exception:  # noqa: BLE001
            response = str(filt)
    else:
        first = (
            args.get("arg_0", {})
            if isinstance(args, dict)
            else (args[0] if args else {})
        )
        prompt = (
            first.get("arg_0", "")
            if isinstance(first, dict)
            else (first[0] if first else "")
        )
        filt = sample.get("filtered_resps") or sample.get("resps") or []
        if filt:
            response = filt[0] if isinstance(filt[0], str) else str(filt[0])

    metadata: dict[str, Any] = {
        "task": task_name,
        "doc_id": sample.get("doc_id"),
        "target": sample.get("target"),
        "filter": sample.get("filter"),
        "output_type": output_type,
    }
    for metric_name in sample.get("metrics", []) or []:
        if metric_name in sample:
            metadata[metric_name] = sample[metric_name]

    return trackio_module.Trace(
        messages=[
            {"role": "user", "content": str(prompt)},
            {"role": "assistant", "content": str(response)},
        ],
        metadata=metadata,
    )


class TrackioLogger:
    def __init__(self, init_args: dict[str, Any] | None = None) -> None:
        """Logs lm-evaluation-harness results to Trackio.

        Trackio (https://github.com/gradio-app/trackio) is a lightweight,
        local-first experiment tracker with a wandb-compatible API plus a
        first-class Trace primitive for conversational/eval samples.

        Args:
            init_args: kwargs forwarded to ``trackio.init``. Supports
                ``project``, ``name``, ``group``, ``space_id``, ``config``,
                and a ``step`` key consumed locally for default log step.

        Usage from the CLI: ``--trackio_args project=my-evals``.

        Programmatic usage::

            trackio_logger.post_init(results)
            trackio_logger.log_eval_result()
            trackio_logger.log_eval_samples(samples)
        """
        try:
            import trackio
        except ImportError as e:
            raise ImportError(
                "trackio is not installed. Install it with `pip install trackio` "
                "or `pip install lm_eval[trackio]`."
            ) from e

        self._trackio = trackio
        self.trackio_args: dict[str, Any] = dict(init_args or {})
        self.step = self.trackio_args.pop("step", None)
        if "project" not in self.trackio_args:
            self.trackio_args["project"] = "lm-eval-harness"
        self.run = trackio.init(**self.trackio_args)

    def post_init(self, results: dict[str, Any]) -> None:
        self.results: dict[str, Any] = copy.deepcopy(results)
        self.task_names: list[str] = list(results.get("results", {}).keys())
        self.group_names: list[str] = list(results.get("groups", {}).keys())
        self.task_configs: dict[str, Any] = self.results.get("configs", {})

    def _flatten_results_for_log(self) -> dict[str, Any]:
        """Flatten nested {task: {metric: value}} into {task/metric: value} scalars."""
        flat: dict[str, Any] = {}
        for task_name, metrics in self.results.get("results", {}).items():
            if not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                key, _ = remove_none_pattern(k)
                if isinstance(v, (int, float)):
                    flat[f"{task_name}/{key}"] = v
        return flat

    def log_eval_result(self) -> None:
        """Log aggregate eval metrics + run config to Trackio."""
        try:
            self._trackio.config.update(
                {
                    "task_configs": self.task_configs,
                    "cli_configs": self.results.get("config", {}),
                }
            )
        except Exception:
            logger.exception("Failed to update Trackio config")

        flat = self._flatten_results_for_log()
        if flat:
            self._trackio.log(flat, step=self.step)

    def log_eval_samples(self, samples: dict[str, list[dict[str, Any]]]) -> None:
        """Log per-sample eval data as Trackio Traces.

        Each sample becomes one ``trackio.Trace`` whose messages capture the
        prompt and (filtered) model response, with the gold target plus all
        per-sample metric values attached as metadata.
        """
        for task_name in self.task_names:
            if task_name in self.group_names or task_name not in samples:
                continue
            task_cfg = self.task_configs.get(task_name, {}) or {}
            output_type = task_cfg.get("output_type", "generate_until")
            for idx, sample in enumerate(samples[task_name]):
                try:
                    trace = _sample_to_trace(
                        self._trackio, sample, task_name, output_type
                    )
                except Exception:
                    logger.exception(
                        "Failed to build trace for sample %s of %s", idx, task_name
                    )
                    continue
                self._trackio.log({f"{task_name}/sample": trace}, step=self.step)

    def finish(self) -> None:
        try:
            self._trackio.finish()
        except Exception:
            logger.exception("Trackio finish raised")
