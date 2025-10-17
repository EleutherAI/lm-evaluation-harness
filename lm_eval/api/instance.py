from dataclasses import dataclass, field
from typing import Any, Literal


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class Instance:
    request_type: OutputType
    doc: dict[str, Any]
    arguments: tuple
    idx: int
    task_name: str
    doc_id: int
    metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata=dict(description="Extra metata can be added here"),
    )
    resps: list = field(
        default_factory=list,
        metadata=dict(
            description="List of responses from the model for this instance."
        ),
    )
    raw_resps: list[str] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    filtered_resps: dict = field(
        default_factory=dict,
        metadata=dict(
            description="List of filtered responses for this instance, keyed by filter name."
        ),
    )
    repeats: int = 1

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self) -> tuple:
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
