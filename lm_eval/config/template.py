from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from lm_eval.config.utils import create_mc_choices


@runtime_checkable
class Template(Protocol):
    format: str
    question: str | None
    choices: list[str] | None
    target: str | int | None
    # Prefix before the question, e.g. "Question: "
    prefix: str
    # Delimiter between question and choices, e.g. "\n"
    question_choice_delimiter: str
    # Suffix after the question, e.g. "\nAnswer:"
    suffix: str
    # Delimiter between choices, e.g. "\n"
    choice_delimiter: str
    # Delimiter between Answer and current choice (for fewshots)
    target_delimiter: str
    # Format of the choices, e.g. "letters" or "numbers"
    choice_format: str | None

    def format_prompt(
        self,
        q: Any,
        c: Any,
        a: Any,
        **kwargs,
    ) -> Any: ...
    def format_choices(
        self,
        q: Any,
        c: Any,
        a: Any,
        **kwargs,
    ) -> Any | None: ...
    def format_target(
        self,
        q: Any,
        c: Any,
        a: Any,
        **kwargs,
    ) -> Any: ...


@dataclass
class MMLUTemplate:
    format: str = "mmlu"
    question: str | None = None
    choices: list[str] | None = None
    target: str | int | None = None
    prefix: str = "Question: "
    question_choice_delimiter: str = "\n"
    suffix: str = "\nAnswer:"
    choice_delimiter: str = "\n"
    target_delimiter: str = "\n\n"
    choice_format: str | None = "letters"

    def format_prompt(
        self,
        q: str,
        c: list[str],
        a: int,
        **kwargs,
    ) -> str:
        return (
            self.prefix
            + q
            + self.question_choice_delimiter
            + create_mc_choices(c, self.choice_delimiter)
            + self.suffix
        )

    def format_choices(
        self,
        q: Any,
        c: list[str],
        a: Any,
        **kwargs,
    ):
        return c

    def format_target(
        self,
        q: Any,
        c: Any,
        a: int,
        **kwargs,
    ):
        return a


@dataclass
class ClozeTemplate:
    format: str = "cloze"
    question: str | None = None
    choices: list[str] | None = None
    target: str | int | None = None
    prefix: str = "Question: "
    question_choice_delimiter: str = "\n"
    suffix: str = "\nAnswer:"
    choice_delimiter: str = ""
    target_delimiter: str = "\n\n"
    choice_format: str | None = ""

    def format_prompt(self, q: str, c: list[str], a: int, **kwargs) -> str:
        return self.prefix + q + self.suffix

    def format_choices(self, q: str, c: list[str], a: int, **kwargs):
        return c

    def format_target(self, q: str, c: list[str], a: int, **kwargs) -> int:
        return a


@dataclass
class GenerateTemplate:
    format: str = "generate_until"
    question: str | None = None
    choices: list[str] | None = None
    target: str | int | None = None
    prefix: str = "Question: "
    question_choice_delimiter: str = "\n"
    suffix: str = "\nAnswer:"
    choice_delimiter: str = "\n"
    target_delimiter: str = "\n\n"
    choice_format: str | None = "letters"

    def format_prompt(
        self,
        q: str,
        c: list[str],
        a: int,
        **kwargs,
    ) -> str:
        return (
            self.prefix
            + q
            + self.question_choice_delimiter
            + create_mc_choices(c, self.choice_delimiter)
            + self.suffix
        )

    def format_choices(self, q: str, c: list[str], a: int, **kwargs):
        return []

    def format_target(self, q: str, c: list[str], a: int, **kwargs):
        return c[a]


def init_template(cfg: str | dict | Template | None):
    """Initialize a template from a string or dict."""
    # fmt: off
    match cfg:
        case Template() | None: return cfg
        case str():
            cfg = {"format": cfg}

    _format = cfg.get("format", "").lower()

    match _format:
        case "mmlu": return MMLUTemplate(**cfg)
        case "cloze": return ClozeTemplate(**cfg)
        case "generate_until": return GenerateTemplate(**cfg)
        case _: raise ValueError(f"Unknown template format: {_format}")
    # fmt: on
