from dataclasses import dataclass, field
from typing import Any, Generic

from typing_extensions import TypedDict, TypeVar

from lm_eval.api._types import GenArgs, GenResponse, LLArgs, LLResponse
from lm_eval.result_schema import OutputType


ArgsT = TypeVar("ArgsT", bound=LLArgs | GenArgs)
RespT = TypeVar("RespT", bound=LLResponse | GenResponse)


class AdditionalArgs(TypedDict, total=False, extra_items=Any):
    """
    Additional arguments that can be passed to the instance, e.g. for generation tasks.
    """

    audio: Any
    visual: Any


@dataclass
class Instance(Generic[ArgsT, RespT]):
    request_type: OutputType
    doc: dict[str, Any]
    arguments: ArgsT
    task_name: str
    doc_id: int = field(kw_only=True)
    idx: int = 0
    repeats: int = 1
    target: str | int | list[str] | list[int] | None = None
    additional_args: AdditionalArgs | None = None
    resps: RespT = field(default_factory=list)
    filtered_resps: dict[str, RespT] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # backward stub for metadata unpacking
    def __post_init__(self) -> None:
        # unpack metadata field
        if isinstance(self.metadata, tuple):
            self.task_name, self.doc_id, self.repeats = self.metadata
            self.metadata = {}

    @property
    def args(self) -> ArgsT:
        """
        Returns (string,) where `string` is the string to calculate loglikelihood over
        """
        return (
            self.arguments if isinstance(self.arguments, tuple) else (self.arguments,)
        )


LLInstance = Instance[LLArgs, LLResponse]
GenInstance = Instance[GenArgs, GenResponse]
