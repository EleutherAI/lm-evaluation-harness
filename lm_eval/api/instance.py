from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple
    idx: int
    metadata: Tuple[Optional[str], Optional[int], Optional[int]] = field(
        default_factory=lambda: (None, None, None),
        metadata=dict(
            description="Metadata tuple containing task name, document ID, and number of repeats."
        ),
    )
    resps: list = field(
        default_factory=list,
        metadata=dict(
            description="List of responses from the model for this instance."
        ),
    )
    filtered_resps: dict = field(
        default_factory=dict,
        metadata=dict(
            description="List of filtered responses for this instance, keyed by filter name."
        ),
    )

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None

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
