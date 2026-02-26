from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic

from typing_extensions import TypedDict, TypeVar

from ._types import Completion, GenArgs, LLArgs, LLOutput


if TYPE_CHECKING:
    from lm_eval.result_schema import OutputType


InputT = TypeVar("InputT", bound=LLArgs | GenArgs)
OutputT = TypeVar("OutputT", bound=list[LLOutput] | list[Completion])


class AdditionalArgs(TypedDict, total=False):
    """Additional arguments that can be passed to the instance, e.g. for multimodal tasks."""

    audio: Any
    visual: Any


@dataclass
class Instance(Generic[InputT, OutputT]):
    request_type: OutputType
    doc: dict[str, Any]
    arguments: InputT
    task_name: str
    doc_id: int = field(kw_only=True)
    idx: int = 0
    repeats: int = 1
    target: str | int | list[str] | list[int] | None = None
    additional_args: AdditionalArgs | None = None
    resps: OutputT = field(default_factory=list)
    filtered_resps: dict[str, OutputT] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # backward stub for metadata unpacking
    def __post_init__(self) -> None:
        # unpack metadata field
        if isinstance(self.metadata, tuple):
            self.task_name, self.doc_id, self.repeats = self.metadata
            self.metadata = {}

    @property
    def args(self) -> InputT:
        """Returns (string,) where `string` is the string to calculate loglikelihood over."""
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )


LLInstance = Instance[LLArgs, list[LLOutput]]
GenInstance = Instance[GenArgs, list[Completion]]
