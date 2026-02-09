from dataclasses import dataclass, field
from typing import Any

from lm_eval.types import OutputType


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple[str, str] | tuple[str, dict[str, Any]]
    idx: int
    metadata: tuple[str | None, int | None, int | None] = field(
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: str | None = None
    doc_id: int | None = None
    repeats: int | None = None

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
