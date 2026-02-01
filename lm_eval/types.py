from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


@dataclass
class MCResult:
    """Result of a multiple-choice task. Instances are grouped by doc_id beforehand"""

    lls: Sequence[float]
    is_greedy: Sequence[bool]
    target: int
    instances: Sequence["Instance"] | None = None
    choices: Sequence[str] = field(default_factory=list)
    char_lens: Sequence[int] = field(default_factory=list)
    byte_lens: Sequence[int] = field(default_factory=list)
    lls_mutual_info: Sequence[float] = field(default_factory=list)
    scores: dict[Any, float] = field(default_factory=dict)

    @classmethod
    def from_instances(cls, results: Sequence["Instance"], acc_mutual_info=False):
        from itertools import chain

        import numpy as np

        ## TODO: ADD Choice/Target Verification
        instance = sorted(
            results,
            key=lambda x: (x.doc_id, x.metadata.get("acc_mutual_info", False)),
        )
        resps, choices, targets = zip(
            *((inst.resps, inst.args[1], inst.target) for inst in instance), strict=True
        )

        lls, is_greedy = zip(*chain.from_iterable(resps), strict=True)
        # Handle mutual information if needed
        lls_mutual_info = []
        if acc_mutual_info:
            assert 2 * len(set(choices)) == len(resps), (
                "Number of results are not equal for acc mutual info; This is unexpected. Please open an issue on github."
            )
            # Then we are doing mutual info.
            # This stores the "dryrun" / unconditional answer loglikelihoods
            # as we extend the args list with unconditional ("", continuation) pairs
            lls_unconditional = lls[len(choices) :]
            if len(lls_unconditional) != len(choices):
                raise ValueError("Number of results are not equal for acc mutual info")
            # And this stores our "regular" conditional loglikelihoods
            lls = lls[: len(choices)]
            lls_mutual_info = [
                ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
            ]

        # calculate lengths
        completion_len = (
            np.array([float(len(i)) for i in choices])
            if choices
            else [1 for _ in range(len(lls))]
        )
        bytes_len = (
            np.array([len(i.encode("utf-8")) for i in choices])
            if choices
            else [1 for _ in range(len(lls))]
        )
        assert len(set(targets)) == 1, (
            "Multiple targets found for same sample; This is unexpected. Please open an issue on github."
        )
        return cls(
            lls=lls,
            is_greedy=is_greedy,
            target=targets[0],
            choices=choices,
            char_lens=completion_len,  # type: ignore
            byte_lens=bytes_len,  # type: ignore
            lls_mutual_info=lls_mutual_info,
            instances=instance,
        )
