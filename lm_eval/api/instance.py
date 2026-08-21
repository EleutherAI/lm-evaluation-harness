from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple


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
        default_factory=lambda: (None, None, None)
    )
    resps: list = field(default_factory=list)
    raw_resps: list[str | None] = field(
        default_factory=list, init=False, repr=False, compare=False
    )
    filtered_resps: dict = field(default_factory=dict)

    # initialized after init
    task_name: Optional[str] = None
    doc_id: Optional[int] = None
    repeats: Optional[int] = None

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

    def append_response(self, response: Any, raw_response: str | None = None) -> None:
        """Append a model response and keep generation metadata request-aligned."""
        if self.request_type != "generate_until":
            self.resps.append(response)
            return

        raw_resps = getattr(self, "raw_resps", None)
        if raw_resps is None:
            raw_resps = []
            self.raw_resps = raw_resps
        previous_response_count = len(self.resps)
        if len(raw_resps) < previous_response_count:
            raw_resps.extend([None] * (previous_response_count - len(raw_resps)))
        elif len(raw_resps) > previous_response_count:
            raise ValueError(
                "Raw generation responses cannot outnumber processed responses."
            )
        self.resps.append(response)
        raw_resps.append(raw_response)

    @property
    def resps_for_logging(self) -> list:
        """Return raw generations when available, otherwise processed responses."""
        raw_resps = getattr(self, "raw_resps", None)
        if not raw_resps:
            return self.resps
        if len(raw_resps) != len(self.resps):
            raise ValueError(
                "Raw and processed generation response counts must remain aligned."
            )
        return [
            raw if raw is not None else processed
            for processed, raw in zip(self.resps, raw_resps, strict=True)
        ]
