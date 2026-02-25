from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import Self


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy import float64, int64
    from numpy._typing import NDArray

    from lm_eval.api.instance import LLInstance


_count_bytes = lambda x: len(x.encode("utf-8"))
_count_words = lambda x: len(re.split(r"\s+", x))


def _empty_array():
    import numpy as np

    return np.array([], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LLResults:
    """Per-doc bundle of log-likelihoods, greedy flags, and choices for loglikelihood tasks.

    Built via :meth:`from_instances` from all ``LLInstance``s sharing a ``doc_id``,
    and passed as ``predictions`` to metrics in :class:`LLScorer`.
    """

    results: list[Any]
    lls: NDArray[float64] = field(kw_only=True)
    is_greedy: Sequence[bool] = field(kw_only=True)
    targets: int | list[int] | str | list[str]
    ctx: str = ""
    choices: Sequence[str] = field(default_factory=list)
    lls_mutual_info: NDArray[float64] = field(default_factory=_empty_array)
    metadata: dict[str, Any] = field(default_factory=dict)

    def char_len(self) -> NDArray[float64]:
        import numpy as np

        return (
            np.array([float(len(i)) for i in self.choices])
            if self.choices
            else np.ones(len(self.lls))
        )

    def byte_len(
        self, count_bytes: Callable[[str], float] = _count_bytes
    ) -> NDArray[int64]:
        import numpy as np

        return np.array(
            [count_bytes(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))],
            dtype=float,
        )

    def word_len(
        self, count_words: Callable[[str], float] = _count_words
    ) -> NDArray[int64]:
        import numpy as np

        return np.array(
            [count_words(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))],
            dtype=float,
        )

    @classmethod
    def from_instances(
        cls,
        results: Sequence[LLInstance],
    ) -> Self:
        from itertools import chain

        import numpy as np

        instances: list[LLInstance] = sorted(
            results,
            key=lambda x: (x.doc_id, x.metadata.get("acc_mutual_info", False)),
        )
        resps, choices, targets, is_mi = zip(
            *(
                (
                    inst.resps,
                    inst.args[1],
                    inst.target,
                    inst.metadata.get("acc_mutual_info", False),
                )
                for inst in instances
            ),
            strict=True,
        )

        lls, is_greedy = zip(*chain.from_iterable(resps), strict=True)
        lls = np.array(lls)

        n_cond = sum(not mi for mi in is_mi)
        if n_cond < len(instances):
            assert 2 * n_cond == len(resps), (
                f"Expected 2 * {n_cond} conditional instances == {len(resps)} total instances "
                "for mutual info. Please open an issue on github."
            )
            # per-element choices should be equal
            # Sort puts conditional instances first. Both sets share the same choice order (see MultipleChoiceTask._create_instances).
            assert choices[:n_cond] == choices[n_cond:], (
                "Conditional/unconditional choice order mismatch"
            )
            # Split: conditional 0..n_cond-1, unconditional n_cond..end
            lls, lls_unconditional = lls[:n_cond], lls[n_cond:]
            is_greedy, choices = is_greedy[:n_cond], choices[:n_cond]
            lls_mutual_info = lls - lls_unconditional
        else:
            lls_mutual_info = _empty_array()

        return cls(
            results=list(resps),
            lls=lls,
            is_greedy=is_greedy,
            ctx=instances[0].args[0],
            targets=targets[0],
            choices=choices,
            lls_mutual_info=lls_mutual_info,
        )

    def to_metric_inputs(self):
        return {"references": self.targets, "predictions": self}
