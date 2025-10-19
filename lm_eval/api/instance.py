from dataclasses import dataclass, field
from typing import Any, Literal


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass(frozen=True, slots=True)
class Instance:
    doc: dict[str, Any]
    arguments: tuple[Any, Any] | Any
    idx: int
    task_name: str
    doc_id: int
    target: Any
    request_type: OutputType
    resps: list = field(
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
    token_len: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata=dict(description="Extra metadata can be added here"),
    )

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )


class MCInstance(Instance):
    arguments: tuple[str, str]
    target: int
    request_type = "loglikelihood"


class GenInstance(Instance):
    arguments: tuple[str, dict[str, Any]] | tuple[list[dict[str, str]], dict[str, Any]]
    target: str
    request_type = "generate_until"
