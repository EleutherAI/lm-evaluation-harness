from typing import Any, Generic, TypeVar

from typing_extensions import TypedDict


T = TypeVar("T", bound=int | float | bool | tuple)


class _TaskMetrics(TypedDict, Generic[T], extra_items=T):
    """Per-task metric dict passed through evaluation and display.

    Fixed keys are ``name``, ``alias`` and ``sample_len``.  The remaining keys are
    dynamic ``metric,filter`` pairs like ``"acc,none"`` and ``"acc_stderr,none"``.
    """

    name: str
    """Name of the Task."""

    alias: str
    """Display name for the task (falls back to task name)."""

    sample_len: int
    """Number of documents evaluated for this task."""


class _SampleCount(TypedDict):
    """Number of evaluation samples for a task."""

    original: int
    """Total number of documents in the evaluation set before any limit is applied."""

    effective: int
    """Actual number of documents evaluated after applying the limit."""


class _EvalConfig(TypedDict, total=False):
    """Model and execution configuration stored in results."""

    model: str
    model_args: str | dict[str, str | int | float] | None
    batch_size: int | str | None
    batch_sizes: list[int]
    device: str | None
    use_cache: str | None
    limit: int | float | None
    bootstrap_iters: int
    gen_kwargs: str | dict | None
    random_seed: int
    numpy_seed: int
    torch_seed: int
    fewshot_seed: int


EvalResults = TypedDict(
    "EvalResults",
    {
        # --- Core evaluation outputs (from evaluate()) ---
        #
        # Per-task metric values.
        "results": dict[str, _TaskMetrics],
        # Aggregated group-level metrics (same shape as "results").
        # Only present when groups are defined
        "groups": dict[str, _TaskMetrics],
        # Maps group/task names to their list of subtask names.
        "group_subtasks": dict[str, list[str]],
        # Full YAML task configs keyed by task name.
        "configs": dict[str, dict[str, Any]],
        # Task version from YAML metadata, keyed by task name.
        "versions": dict[str, str | float | None],
        # Number of few-shot examples used per task.
        "n-shot": dict[str, int],
        # Per-task dict mapping metric name to whether higher is better.
        "higher_is_better": dict[str, dict[str, bool | None]],
        # Original and effective (after limit) sample counts per task.
        "n-samples": dict[str, _SampleCount],
        # Per-task list of per-document log dicts.
        # Only present when log_samples is True.
        "samples": dict[str, list[dict[str, Any]]],
        # --- Metadata added by simple_evaluate() ---
        #
        # Model and execution configuration.
        "config": _EvalConfig,
        # Git commit hash at evaluation time.
        "git_hash": str,
        # UNIX timestamp when evaluation started.
        "date": float,
        # --- Environment info (added by add_env_info()) ---
        #
        # PyTorch environment info string.
        "pretty_env_info": str,
        # Installed transformers library version.
        "transformers_version": str,
        # Installed lm_eval library version.
        "lm_eval_version": str,
        # Git hash of the parent repo, if this repo is a submodule.
        "upper_git_hash": str | None,
        # --- Tokenizer info (added by add_tokenizer_info()) ---
        #
        # Pad token and its ID as [token, token_id].
        # [lm.tokenizer.pad_token, str(lm.tokenizer.pad_token_id)]
        "tokenizer_pad_token": list[str],
        # EOS token and its ID as [token, token_id].
        # [lm.tokenizer.eos_token, str(lm.tokenizer.eos_token_id)]
        "tokenizer_eos_token": list[str],
        # BOS token and its ID as [token, token_id].
        # [lm.tokenizer.bos_token, str(lm.tokenizer.bos_token_id)]
        "tokenizer_bos_token": list[str],
        # End-of-text token ID used by the model.
        # getattr(lm, "eot_token_id", None)
        "eot_token_id": int | None,
        # Maximum sequence length used for evaluation.
        # getattr(lm, "max_length", None) inside add_tokenizer_info()
        "max_length": int | None,
    },
    total=False,
)
"""Full evaluation results returned by ``simple_evaluate()`` and ``evaluate()``.

All keys are optional (``total=False``) because several are conditionally present:

- ``groups`` — only present when at least one group defines aggregation metrics
- ``samples`` — only when log_samples is True
- ``config``, ``git_hash``, ``date``, env/tokenizer info — only added by
  simple_evaluate()
"""
