from collections.abc import Sequence  # noqa: TC003
from typing import TYPE_CHECKING, Any, TypeAlias
from typing_extensions import Protocol, TypedDict


if TYPE_CHECKING:
    from collections.abc import Mapping

    import datasets


Doc = dict[str, Any]
"""A single document (row) from the dataset, mapping field names to values."""

DataSplit: TypeAlias = "datasets.Dataset | Sequence[Doc]"
"""A single dataset split — iterable + sized collection of docs."""

Dataset: TypeAlias = "Mapping[str, DataSplit] | datasets.DatasetDict"
"""The full dataset — maps split names to splits."""

Context = str | list[dict[str, str]]
"""The context passed to the model: a string or list of chat messages."""


LLArgs = tuple[str, str]
"""Arguments for a loglikelihood request: ``(context, continuation)``."""

LLOutput = tuple[float, bool]
"""Output of a single loglikelihood request: ``(logprob, is_greedy)``."""

GenArgs = tuple[Context, "GenKwargs"]
"""Arguments for a generation request: ``(context, gen_kwargs)``."""

Completion = str
"""Output of a single generation request."""

Reference = str | list[str] | int | list[int] | None
"""Gold-standard reference for a document."""


class ChatTemplate(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: Sequence[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Context: ...


class GenKwargs(TypedDict, total=False):
    """Keyword arguments controlling text generation."""

    do_sample: bool
    temperature: float
    # other alias' will be converted to `max_gen_toks`.
    max_gen_toks: int
    until: list[str]
    __extra_items__: Any
