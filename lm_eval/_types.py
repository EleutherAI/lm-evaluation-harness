from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import Protocol


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import float64, int64
    from numpy.typing import NDArray

    from lm_eval.api.instance import Instance

_count_bytes = lambda x: len(x.encode("utf-8"))
_count_words = lambda x: len(re.split(r"\s+", x))


class ChatTemplate(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str | list[dict[str, Any]]: ...


# multiple_choice types send a number of "loglikelihood" instances
OutputType = Literal["loglikelihood", "loglikelihood_rolling", "generate_until"]


@dataclass(frozen=True, slots=True)
class GenResults:
    ctx: str
    targets: list[str]
    results: list[dict[str, list[str]]]
    doc: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_instances(cls, results: Sequence[Instance]):
        instance: list[Instance] = sorted(results, key=lambda x: x.doc_id)
        targets = [inst.target for inst in instance]
        _results = [i.filtered_resps for i in instance]
        ctx = instance[0].args[0] if instance else ""
        return cls(doc=instance[0].doc, ctx=ctx, targets=targets, results=_results)

    def to_metric_inputs(self):
        return {"references": self.targets, "predictions": self.results}


@dataclass(frozen=True, slots=True)
class LLResults:
    """Result of a multiple-choice task. Instances are grouped by doc_id beforehand"""

    results: list[str] | list[list[tuple[float, bool]]]
    targets: int | list[int] | str | list[str]
    ctx: str = ""
    doc: dict[str, Any] = field(default_factory=dict)
    choices: Sequence[str] = field(default_factory=list)
    lls: NDArray[float64] = field(kw_only=True)
    is_greedy: Sequence[bool] = field(kw_only=True)
    lls_mutual_info: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # @property
    # def target(self) -> int:
    #     return self.targets[0] if isinstance(self.targets, list) else self.targets

    def char_len(self) -> NDArray[float64]:
        import numpy as np

        return (
            np.array([float(len(i)) for i in self.choices])
            if self.choices
            else np.array(1.0 for _ in range(len(self.lls)))
        )

    def byte_len(self) -> NDArray[int64]:
        import numpy as np

        return np.array(
            [_count_bytes(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))]
        )

    def word_len(self) -> NDArray[int64]:
        import numpy as np

        return np.array(
            [_count_words(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))]
        )

    @classmethod
    def from_instances(
        cls,
        results: Sequence[Instance],
    ):
        from itertools import chain

        instance = sorted(
            results,
            key=lambda x: (x.doc_id, x.metadata.get("acc_mutual_info", False)),
        )
        resps, choices, targets = zip(
            *((inst.resps, inst.args[1], inst.target) for inst in instance), strict=True
        )

        lls, is_greedy = zip(*chain.from_iterable(resps), strict=True)
        lls = np.array(lls)
        targets = list(set(targets))[0]
        # Handle mutual information if needed
        if 2 * len(set(choices)) == len(resps):
            # Then we are doing mutual info.
            # This stores the "dryrun" / unconditional answer loglikelihoods
            # as we extend the args list with unconditional ("", continuation) pairs
            lls_unconditional = lls[len(choices) :]
            if len(lls_unconditional) != len(choices):
                raise ValueError("Number of results are not equal for acc mutual info")
            # And this stores our "regular" conditional loglikelihoods
            lls = lls[: len(choices)]
            lls_mutual_info = lls - lls_unconditional

            # TODO: fix
            # assert len(set(targets)) == 1, (
            #     "Multiple targets found for same sample; This is unexpected. Please open an issue on github."
            # )

        else:
            lls_mutual_info = np.array([], dtype=np.float64)

        return cls(
            results=[inst.resps for inst in instance],
            lls=lls,
            is_greedy=is_greedy,
            ctx=instance[0].args[0],
            targets=targets,
            choices=choices,
            lls_mutual_info=lls_mutual_info,
        )

    def to_metric_inputs(self):
        return {"references": self.targets, "predictions": self}
