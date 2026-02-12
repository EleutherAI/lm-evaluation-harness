from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, TypedDict


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


# input arguments for loglikelihood and multiple-choice tasks
class LLArgs(NamedTuple):
    ctx: str
    cont: str


# input arguments for loglikelihood rolling tasks
class LLRollingArgs(NamedTuple):
    ctx: str
    cont: None


# arguments for generation tasks
class GenArgs(NamedTuple):
    prompt: str | list[dict[str, str]]
    gen_kwargs: "GenKwargs"


class MultiModalArgs(TypedDict, total=False):
    image: dict[str, Any]
    audio: dict[str, Any]
    video: dict[str, Any]


class GenKwargs(TypedDict, total=False):
    # total number of tokens to generate
    max_gen_toks: int
    # weather to use sampling
    do_sample: bool
    # list of strings to stop generation at
    temperature: float
    until: list[str]
    multimodal_args: "MultiModalArgs"
    # top_p, top_k, repetition_penalty, etc.


@dataclass(frozen=True, slots=True)
class Instance:
    request_type: OutputType
    doc: dict
    arguments: LLArgs | LLRollingArgs | GenArgs
    idx: int
    task_name: str
    doc_id: int
    target: str | int | None = None
    metadata: tuple[str | None, int | None, int | None] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    raw_resps: list = field(default_factory=list)
    filtered_resps: dict[str, list[str]] = field(default_factory=dict)

    repeats: int = 1

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )


@dataclass(frozen=True, slots=True)
class MCInstance(Instance):
    """Multiple choice / loglikelihood instance with integer target."""

    arguments: LLArgs = field(kw_only=True)
    target: int = field(kw_only=True)
    request_type: Literal["loglikelihood"] = field(
        default="loglikelihood", kw_only=True
    )
    resps: list[tuple[float, bool]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GenInstance(Instance):
    """Generation instance with a string target."""

    arguments: GenArgs = field(kw_only=True)
    target: str = field(kw_only=True)
    request_type: Literal["generate_until"] = field(
        default="generate_until",
        kw_only=True,
    )
    resps: list[str] = field(default_factory=list)


@dataclass
class ListInstance:
    instances: list[Instance]
    _scores: list[float] = field(default_factory=list)
