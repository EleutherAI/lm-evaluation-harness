from typing import Any, Generic, TypeVar

from typing_extensions import NotRequired, TypedDict


T = TypeVar("T", bound=int | float | bool | tuple)


"""Full evaluation results returned by ``simple_evaluate()`` and ``evaluate()``.

All keys are optional (``total=False``) because several are conditionally present:

- ``groups`` — only present when at least one group defines aggregation metrics
- ``samples`` — only when log_samples is True
- ``config``, ``git_hash``, ``date``, env/tokenizer info — only added by
  simple_evaluate()
"""

EvalResults = TypedDict(
    "EvalResults",
    {
        # --- Core evaluation outputs (from evaluate()) ---
        #
        # Per-task metric values.
        "results": "dict[str, _TaskMetrics]",
        # Aggregated group-level metrics (same shape as "results").
        # Only present when groups are defined
        "groups": "dict[str, _TaskMetrics]",
        # Maps group/task names to their list of subtask names.
        "group_subtasks": dict[str, list[str]],
        # Full YAML task configs keyed by task name.
        "configs": dict[str, dict[str, str | bool | None]],
        # Task version from YAML metadata, keyed by task name.
        "versions": dict[str, str | float | None],
        # Number of few-shot examples used per task.
        "n-shot": dict[str, int],
        # Per-task dict mapping metric name to whether higher is better.
        "higher_is_better": dict[str, dict[str, bool | None]],
        # Original and effective (after limit) sample counts per task.
        "n-samples": "dict[str, _SampleCount]",
        # Per-task list of per-document sample results.
        # Only present when log_samples is True.
        "samples": "dict[str, list[SampleResult]]",
        # --- Metadata added by simple_evaluate() ---
        #
        # Model and execution configuration.
        "config": "_EvalConfig",
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
        # --- Model identity (added by simple_evaluate()) ---
        #
        # Model source identifier (e.g. "hf").
        "model_source": str,
        # Full model name (e.g. "EleutherAI/pythia-14m").
        "model_name": str,
        # Sanitized model name safe for file paths.
        "model_name_sanitized": str,
        # --- Chat / instruction fields ---
        #
        # System instruction passed to the model, if any.
        "system_instruction": str | None,
        # SHA of the system instruction.
        "system_instruction_sha": str | None,
        # Whether few-shot examples were formatted as multi-turn.
        "fewshot_as_multiturn": bool | None,
        # Chat template string, if applicable.
        "chat_template": str | None,
        # SHA of the chat template.
        "chat_template_sha": str | None,
        # --- Misc metadata ---
        #
        # Per-task hash values for reproducibility verification.
        "task_hashes": dict[str, str],
        # Wall-clock evaluation time in seconds (stored as string).
        "total_evaluation_time_seconds": str,
    },
    total=False,
)


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

    sample_count: NotRequired[dict[str, int]]
    """Per-metric sample counts (groups only). Maps metric keys like
    ``"acc,none"`` to the number of samples used for that metric."""


class _SampleCount(TypedDict):
    """Number of evaluation samples for a task."""

    original: int
    """Total number of documents in the evaluation split."""

    effective: int
    """Actual number of documents actually evaluated (e.g. using limit)."""


class _EvalConfig(TypedDict, total=False):
    """Model and execution configuration stored in results."""

    model: str
    model_args: str | dict[str, str | int | float] | None
    model_num_parameters: int
    model_dtype: str
    model_revision: str
    model_sha: str
    batch_size: str | None
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


class SampleResult(TypedDict, extra_items=float):
    """Per-document result written to ``samples_*.jsonl`` when ``log_samples=True``.

    There is one entry per filter (e.g. ``"none"``, ``"strict-match"``).
    Fixed keys are common across all task types (multiple-choice, generation, etc.).
    Dynamic keys are per-sample metric scores (e.g. ``acc``, ``acc_norm``,
    ``exact_match``) whose names and count vary by task — these are always floats.
    """

    doc_id: int
    """Zero-based index of the document within the evaluation split."""

    doc: dict[str, Any]
    """Original document from the dataset for this sample."""

    target: str
    """Gold-standard target string."""

    arguments: dict[str, dict[str, Any]]
    """Per-request model inputs, as ``{"gen_args_N": {"arg_0": ..., "arg_1": ...}}``.
    Multiple-choice: one entry per choice, ``{"arg_0": context, "arg_1": continuation}``.
    Generation: single entry, ``{"arg_0": prompt, "arg_1": gen_kwargs}``."""

    resps: list[list[str]] | list[list[list[str]]]
    """Raw model responses.  Outer list is per-request (one per ``gen_args_N``).
    Generation: ``list[list[str]]`` — requests × repeats × generated text.
    Multiple-choice: ``list[list[list[str]]]`` — requests × repeats × ``[log_prob, is_greedy]``."""

    filtered_resps: list[str] | list[list[str]]
    """Responses after filter application.  Per-request.
    Generation: ``list[str]``.
    Multiple-choice: ``list[list[str]]`` — per-choice ``[log_prob, is_greedy]``."""

    filter: str
    """Name of the filter applied (e.g. ``"none"``, ``"strict-match"``)."""

    metrics: list[str]
    """Names of metrics computed for this sample (e.g. ``["acc", "acc_norm"]``)."""

    doc_hash: str
    """SHA hash of the document for reproducibility verification."""

    prompt_hash: str
    """SHA hash of the prompt sent to the model."""

    target_hash: str
    """SHA hash of the target string."""
