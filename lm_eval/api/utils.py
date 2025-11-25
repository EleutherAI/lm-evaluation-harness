from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def maybe_delimit(prefix: str | None, suffix: str | None, delimiter: str = " ") -> str:
    if not prefix:
        return suffix or ""
    if not suffix:
        return prefix or ""
    return (
        prefix + delimiter + suffix
        if not (prefix[-1].isspace() or suffix[0].isspace())
        else prefix + suffix
    )


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str
    _delimiter: str = ""

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_text(self):
        return self.content + self._delimiter


def messages_to_text(messages: list[Message]) -> str:
    return "".join(m.to_text() for m in messages)


def multiturn_to_singleturn(messages: list[Message]) -> list[dict[str, Any]]:
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
    return (
        {"role": role, "content": content}
        if not type
        else {"type": type, "role": role, "content": content}
    )

def handle_multiple_inputs(q,c,a):
    if not isinstance(q, list):
        return q,c,a

