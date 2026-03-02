from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from lm_eval.api._types import Completion, LLOutput
    from lm_eval.api.instance import Instance

# Response element type: LLProb for loglikelihood, Completion (str) for generation.
_T = TypeVar("_T", "LLOutput", "Completion", default="Completion")


@runtime_checkable
class Filter(Protocol[_T]):
    """Post-process model responses for a task before scoring.

    Filters transform raw model outputs (instance.resps) into a form suitable for metric
    computation.  They operate on **all docs of a task at once**,
    receiving a 2-D structure:

    - outer (Iterable) — one entry per doc
    - inner (Sequence) — one entry per repeat of that doc

    Multiple filters can be chained via [FilterEnsemble][FilterEnsemble].
    ``T`` is the response element type:
    ``Completion`` (``str``) for generation tasks,
    ``LLOutput`` (``tuple[float, bool]``) for loglikelihood tasks.

    Defaults to ``Completion``.
    """

    def apply(
        self,
        resps: Iterable[Sequence[_T]],
        docs: Sequence[dict[str, Any]],
    ) -> Iterable[Sequence[_T]]:
        """Transform model responses.

        Args:
            resps: Per-doc response sequences.  Outer ``Iterable``
                iterates over docs; inner ``Sequence`` holds repeats.
            docs: The source document for each entry (parallel to *resps*).

        Returns:
            Transformed responses **in the same doc order**.  May be
            lazy (``map``) to allow generator chaining between filters.
        """
        ...


@dataclass
class FilterEnsemble:
    """A named chain of [Filter][Filter] steps applied sequentially.

    Each [Scorer][lm_eval.scorers.Scorer] owns one ``FilterEnsemble``.  When applied, it
    extracts ``(resps, doc)`` pairs from every ``Instance``, threads them
    through each filter in order (outputs feed into the next filter's
    inputs), and stores the final result in
    ``Instance.filtered_resps[self.name]``.

    Filters in the chain may return lazy iterables (e.g. ``map``);
    materialisation is deferred until the final ``zip`` writes results back.
    """

    name: str
    filters: list[
        Callable[[], Filter]
    ]  # factories; typically partial(FilterCls, **kwargs) via build_filter_ensemble()

    def apply(self, instances: Sequence[Instance]) -> None:
        resps, docs = zip(*((inst.resps, inst.doc) for inst in instances), strict=True)
        # resps, docs = list(resps), list(docs)

        for f in self.filters:
            # apply filters in sequence
            resps = f().apply(resps, docs)

        # add the end results after filtering to filtered_requests of their respective source instances.
        # has key `self.name`: each FilterEnsemble applied in a given run should use a unique name.
        for inst, resp in zip(instances, resps, strict=True):
            inst.filtered_resps[self.name] = resp
