from dataclasses import dataclass
from typing import Any, TypeGuard

from lm_eval.defaults import DATA_VALIDATION


def is_str_list(choices: list[str] | list[int]) -> TypeGuard[list[str]]:
    return isinstance(choices[0], str) if choices else False


def is_int_list(choices: list[str] | list[int]) -> TypeGuard[list[int]]:
    return isinstance(choices[0], int) if choices else False


def extract_target_index(
    choices: list[str],
    target: str | int | list[int] | list[str],
    _raise: bool = DATA_VALIDATION,
) -> int:
    match target:
        case str() if target in choices:
            return choices.index(target)
        case int() if 0 <= target < len(choices):
            return target
        case list():
            if is_int_list(target):
                for i in target:
                    if 0 <= i < len(choices):
                        return i
            elif is_str_list(target):
                for i in target:
                    if i in choices:
                        return choices.index(i)
    _raise = DATA_VALIDATION
    if _raise:
        raise ValueError(f"Invalid choice: {target}; choices: {choices}")
    return -1


def check_gold_index_error(
    choices: list[int] | list[str], gold: list[int] | int | str
) -> tuple[int | list[int], bool]:
    gold_index_error = False
    if isinstance(gold, list):
        gold = [i if i < len(choices) else -100 for i in gold]
        if -100 in gold:
            gold_index_error = True
            return gold, gold_index_error
    else:
        if isinstance(gold, int):
            gold = gold if gold < len(choices) else -100
        elif isinstance(gold, str):
            gold = choices.index(gold) if gold in choices else -100

        if gold == -100:
            gold_index_error = True
    return gold, gold_index_error


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
