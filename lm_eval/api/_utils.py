from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import TypeVar

from lm_eval import utils


if TYPE_CHECKING:
    from collections.abc import Iterable

    from lm_eval.api.instance import Instance
    from lm_eval.config.task import Dataset


_T = TypeVar("_T")

eval_logger = logging.getLogger(__name__)


def maybe_delimit(prefix: str | None, suffix: str | None, delimiter: str = " ") -> str:
    """Join prefix and suffix, adding delimiter only if neither has whitespace at the boundary."""
    if not prefix:
        return suffix or ""
    if not suffix:
        return prefix or ""
    return (
        prefix + delimiter + suffix
        if not (prefix[-1].isspace() or suffix[0].isspace())
        else prefix + suffix
    )


def requires_delimiter(prefix: str, suffix: str) -> bool:
    """Return True if neither string has whitespace at the join boundary."""
    if prefix == "":
        return False
    return not (prefix[-1].isspace() or suffix[0].isspace())


def ends_with_whitespace(s: str) -> bool:
    """Return True if the string ends with whitespace."""
    return bool(s) and s[-1].isspace()


@dataclass
class Message:
    r"""A single message in a prompt, supporting both chat and plain-text formats.

    Used by `build_qa_turn()` and `fewshot_context()` to construct prompts that can
    be rendered either as plain text or as chat-formatted messages.

    Attributes:
        role (str): "system", "user", or "assistant".
        content (str): The message text.
        _delimiter (str): Suffix appended when rendering as plain text (via `to_text()`).
            Prefixed with `_` so it's excluded from the ` to_dict () ` output. Typically used
            for target delimiters (e.g., " ") or fewshot delimiters (e.g., "\n\n").
    """

    role: str  # "system" | "user" | "assistant"
    content: str
    _delimiter: str = ""

    def to_dict(self) -> dict[str, str]:
        """Convert to chat format dict, excluding internal fields like `_delimiter`."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_text(self) -> str:
        """Render as plain text with delimiter appended."""
        return self.content + self._delimiter


def messages_to_text(messages: list[Message]) -> str:
    """Concatenate all messages into plain text, using each message's `_delimiter`."""
    return "".join(m.to_text() for m in messages)


def multiturn_to_singleturn(messages: list[Message]) -> list[dict[str, str]]:
    """Collapse multi-turn messages into a single user message (plus optional system/assistant)."""
    system, messages = (
        (messages[0], messages[1:])
        if messages[0].role == "system"
        else (None, messages)
    )

    if messages[-1].role == "assistant":
        _user_message = messages[:-1]
        _user_message[-1]._delimiter = ""
        res = [
            Message("user", "".join(m.to_text() for m in _user_message), "").to_dict()
        ] + [messages[-1].to_dict()]
    else:
        messages[-1]._delimiter = ""
        res = [Message("user", "".join(m.to_text() for m in messages), "").to_dict()]

    return [system.to_dict()] + res if system else res


def format_turn(content: str, role: str, _type: str | None = None) -> dict[str, str]:
    """Create a chat message dict with role, content, and optional type."""
    return (
        {"role": role, "content": content}
        if not _type
        else {"type": _type, "role": role, "content": content}
    )


def random_task_id():
    """Generate a random 8-character alphanumeric task ID."""
    import random
    import string

    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def normalize_to_list(x: dict[str, _T | list[_T]]) -> dict[str, list[_T]]:
    """Normalize a dict of str to T or list[T] into a dict of str to list[T]."""
    return {k: v if isinstance(v, list) else [v] for k, v in x.items()}


def _resolve_dataset_path(path: str) -> str:
    """Resolve a HF Hub dataset ID to a local path if ``LM_EVAL_DATASET_DIR`` is set.

    When the env var points to a root directory containing cloned HF dataset
    repos (preserving the ``org/repo`` structure), any matching local directory
    is used instead of fetching from the Hub.

    Example:
        ```bash
        export LM_EVAL_DATASET_DIR=/data/datasets
        # /data/datasets/allenai/ai2_arc/  exists locally
        ```

        ```python
        _resolve_dataset_path("allenai/ai2_arc")
        # -> "/data/datasets/allenai/ai2_arc"
        ```
    """
    import os
    from pathlib import Path

    dataset_dir = os.environ.get("LM_EVAL_DATASET_DIR")
    if dataset_dir:
        local = Path(dataset_dir) / path
        if local.is_dir():
            return str(local)
        else:
            eval_logger.warning(
                "LM_EVAL_DATASET_DIR is set to '%s' but '%s' was not found locally. "
                "Falling back to HF Hub.",
                dataset_dir,
                local,
            )
    return path


def load_dataset_splits(
    path: str,
    name: str | None,
    split: Iterable[str | None] | None = None,
    **kwargs,
) -> Dataset:
    """Load only the specified splits, returning a DatasetDict.

    Accepts any iterable of split names. Duplicates are removed.
    """
    import datasets

    from lm_eval.defaults import LIMIT_DF

    path = _resolve_dataset_path(path)
    _limit = LIMIT_DF if LIMIT_DF.isdigit() else ""
    unique_splits = list({s for s in split if s is not None}) if split else None
    # if split != None, load_dataset returns a list of Datasets, otherwise, it returns a single DatasetDict
    loaded = datasets.load_dataset(
        path,
        name,
        split=[f"{x}[:{_limit}]" for x in unique_splits] if unique_splits else None,
        **kwargs,
    )
    return (
        datasets.DatasetDict(dict(zip(unique_splits, loaded, strict=True)))
        if unique_splits
        else loaded
    )


def _build_cache_key(
    task: str,
    num_fewshot: int | None,
    rank: int,
    world_size: int,
    apply_chat_template: bool,
    fewshot_as_multiturn: bool,
    system_instruction: str | None,
    tokenizer_name: str,
) -> str:
    """Build the cache key string for request caching."""
    parts = [
        f"requests-{task}-{num_fewshot}shot-rank{rank}-world_size{world_size}",
    ]
    if apply_chat_template:
        parts.append("chat_template")
    if fewshot_as_multiturn:
        parts.append("fewshot_as_multiturn")
    if system_instruction is not None:
        parts.append(f"system_prompt_hash{utils.hash_string(system_instruction)}")
    parts.append(f"tokenizer{tokenizer_name}")
    return "-".join(parts)


def group_by_doc_id(
    instances: list[Instance] | None = None,
) -> dict[int, list[Instance]]:
    """Group instances by doc_id and sort each group by idx."""
    if not instances:
        return {}
    from collections import defaultdict

    instances_by_doc_id: dict[int, list[Instance]] = defaultdict(list)
    for instance in instances:
        instances_by_doc_id[instance.doc_id].append(instance)
    for insts in instances_by_doc_id.values():
        insts.sort(key=lambda x: x.idx)
    return instances_by_doc_id
