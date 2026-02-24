from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

from typing_extensions import Protocol, TypedDict


if TYPE_CHECKING:
    from collections.abc import Mapping

    import datasets


Doc = dict[str, Any]
# A single dataset split – iterable + sized collection of docs.
# datasets.Dataset is the primary impl; list[Doc] works for custom datasets.
DataSplit: TypeAlias = "datasets.Dataset | Sequence[Doc]"

# The full dataset – maps split names (training_split, test_split, etc.) to splits.
# datasets.DatasetDict is the primary impl; dict[str, DataSplit] works too.
Dataset: TypeAlias = "Mapping[str, DataSplit] | datasets.DatasetDict"

# the context passed to the model. Usually a string in most cases, but can be a dict of turn-level strings,
# for model implementations process them internally, depending on the chat template used.
Context = str | dict[str, str]


LLArgs = tuple[str, str]
LLResponse = Sequence[tuple[float, bool]]

GenArgs = tuple[Context, "GenKwargs"]
GenResponse = Sequence[str]


class ChatTemplate(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs,
    ) -> Context: ...


class GenKwargs(TypedDict, total=False):
    do_sample: bool
    temperature: float
    # other alias' will be converted to `max_gen_toks`.
    max_gen_toks: int
    until: list[str]
    __extra_items__: Any
