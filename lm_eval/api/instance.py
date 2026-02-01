from dataclasses import dataclass, field
from typing import Literal


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
    metadata: tuple[str | None, int | None, int] = field(
        default_factory=lambda: (None, None, 1)
    )

    # initialized after init
    task_name: str | None = None
    doc_id: int | None = None
    repeats: int = 1

    def __post_init__(self) -> None:
        # unpack metadata field
        self.task_name, self.doc_id, self.repeats = self.metadata

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
