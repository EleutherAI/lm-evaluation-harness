from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict

from lm_eval.types import OutputType


class AdditionalArgs(TypedDict, total=False, extra_items=Any):
    """
    Additional arguments that can be passed to the instance, e.g. for generation tasks.
    """

    audio: Any
    visual: Any


@dataclass
class Instance:
    request_type: OutputType
    doc: dict
    arguments: tuple[str, str] | tuple[str, dict[str, Any]]
    idx: int
    task_name: str
    doc_id: int
    repeats: int = 1
    target: str | int | list[str] | list[int] | None = None
    additional_args: AdditionalArgs | None = None
    resps: list = field(default_factory=list)
    filtered_resps: dict = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # backward stub for metadata unpacking
    def __post_init__(self) -> None:
        # unpack metadata field
        if isinstance(self.metadata, tuple):
            self.task_name, self.doc_id, self.repeats = self.metadata
            self.metadata = {}

    @property
    def args(self):
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )
