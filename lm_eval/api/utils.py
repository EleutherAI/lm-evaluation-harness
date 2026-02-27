from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def multiturn_to_singleturn(messages: list[Message]) -> list[dict[str, Any]]:
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


def format_turn(content: str, role: str, type: str | None = None) -> dict[str, str]:
    """Create a chat message dict with role, content, and optional type."""
    return (
        {"role": role, "content": content}
        if not type
        else {"type": type, "role": role, "content": content}
    )


def random_task_id():
    """Generate a random 8-character alphanumeric task ID."""
    import random
    import string

    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
