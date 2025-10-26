from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from lm_eval.types import GenArgs, LLArgs, LLRollingArgs, MultiModalArgs


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


class TokenLen(TypedDict, total=False):
    ctx: int
    cont: int
    total: int


@dataclass(frozen=True, slots=True)
class Instance:
    """Base class for evaluation instances.

    Use MCInstance for multiple choice/loglikelihood tasks,
    or GenInstance for generation tasks.
    """

    doc: dict[str, Any]
    arguments: LLArgs | GenArgs | LLRollingArgs
    idx: int
    task_name: str
    doc_id: int
    target: int | str
    request_type: OutputType
    multimodal_args: MultiModalArgs = field(default_factory=dict)  # type: ignore
    resps: list[str] | list[tuple[float, bool]] | list[tuple[float, None]] = field(
        default_factory=list,
        metadata=dict(
            description="List of responses from the model for this instance."
        ),
    )
    repeats: int = 1
    raw_resps: list[str] = field(default_factory=list)
    filtered_resps: dict = field(
        default_factory=dict,
        metadata=dict(
            description="List of filtered responses for this instance, keyed by filter name."
        ),
    )
    token_len: TokenLen = field(
        default_factory=lambda: TokenLen(ctx=0, cont=0),
        metadata=dict(description="tokens lengths can be added here after inference"),
    )
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata=dict(description="Extra metadata can be added here"),
    )

    @property
    def args(self) -> LLArgs | GenArgs | LLRollingArgs:
        """
        Returns (context, continuation) for loglikelihood instances,
        or (prompt, None) for generation instances.
        """
        return self.arguments

    @property
    def choice(self):
        return self.arguments[1]


@dataclass(frozen=True, slots=True)
class MCInstance(Instance):
    """Multiple choice / loglikelihood instance with integer target."""

    arguments: LLArgs
    target: int
    request_type: Literal["loglikelihood"] = field(default="loglikelihood")
    resps: list[tuple[float, bool]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GenInstance(Instance):
    """Generation instance with a string target."""

    arguments: GenArgs
    target: str
    request_type: Literal["generate_until"] = field(
        default="generate_until",
    )
    resps: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LLRollingInstance(Instance):
    """Loglikelihood rolling instance with a string target."""

    arguments: LLRollingArgs
    target: str
    request_type: OutputType = field(default="loglikelihood_rolling")
    resps: list[tuple[float, Any]] = field(default_factory=list)
