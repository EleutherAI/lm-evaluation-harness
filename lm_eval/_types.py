import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import NamedTuple, Protocol, Self, TypedDict


if TYPE_CHECKING:
    from numpy import float64, int64
    from numpy.typing import NDArray

    from lm_eval.api.instance import GenInstance, Instance


ChatFormat = str | list[dict[str, Any]]
_count_bytes = lambda x: len(x.encode("utf-8"))
_count_words = lambda x: len(re.split(r"\s+", x))


class ChatTemplateProtocol(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs,
    ) -> ChatFormat: ...


# input arguments for loglikelihood and multiple-choice tasks
class LLArgs(NamedTuple):
    ctx: str
    cont: str


# input arguments for loglikelihood rolling tasks
class LLRollingArgs(NamedTuple):
    ctx: str
    cont: None


# arguments for generation tasks
class GenArgs(NamedTuple):
    prompt: str | list[dict[str, str]]
    gen_kwargs: "GenKwargs"


class MultiModalArgs(TypedDict, total=False):
    image: dict[str, Any]
    audio: dict[str, Any]
    video: dict[str, Any]


class GenKwargs(TypedDict, total=False):
    # total number of tokens to generate
    max_gen_toks: int
    # weather to use sampling
    do_sample: bool
    # list of strings to stop generation at
    temperature: float
    until: list[str]
    multimodal_args: "MultiModalArgs"
    # top_p, top_k, repetition_penalty, etc.


class Results(Protocol):
    def to_metric_inputs(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class LLResults(Results):
    """Result of a multiple-choice task. Instances are grouped by doc_id beforehand"""

    doc: dict[str, Any]
    ctx: str
    targets: int | list[int]
    results: list[str] | None = None
    lls: Sequence[float] = field(kw_only=True)
    is_greedy: Sequence[bool] = field(kw_only=True)
    choices: Sequence[str] = field(default_factory=list)
    # token_lens: Sequence[int] = field(default_factory=list)
    lls_mutual_info: Sequence[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target(self) -> int:
        return self.targets[0] if isinstance(self.targets, list) else self.targets

    @property
    def char_len(self) -> "NDArray[float64]":
        import numpy as np

        return (
            np.array([float(len(i)) for i in self.choices])
            if self.choices
            else np.array(1.0 for _ in range(len(self.lls)))
        )

    @property
    def byte_len(self) -> "NDArray[int64]":
        import numpy as np

        return np.array(
            [_count_bytes(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))]
        )

    @property
    def word_len(self) -> "NDArray[int64]":
        import numpy as np

        return np.array(
            [_count_words(i) for i in self.choices]
            if self.choices
            else [1 for _ in range(len(self.lls))]
        )

    @classmethod
    def from_instances(
        cls,
        results: Sequence["Instance"],
        acc_mutual_info=False,
    ):
        from itertools import chain

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

        assert len(set(targets)) == 1, (
            "Multiple targets found for same sample; This is unexpected. Please open an issue on github."
        )
        return cls(
            doc=instance[0].doc,
            lls=lls,
            is_greedy=is_greedy,
            ctx=instance[0].args[0],
            targets=targets,
            choices=choices,
            lls_mutual_info=lls_mutual_info,
        )

    def to_metric_inputs(self) -> Self:
        return self


@dataclass(frozen=True, slots=True)
class GenResults:
    doc: dict[str, Any]
    ctx: str
    targets: list[str]
    results: list[dict[str, list[str]]]

    @classmethod
    def from_instances(cls, results: Sequence["GenInstance"]):
        instance: list[GenInstance] = sorted(results, key=lambda x: x.doc_id)
        targets = [inst.target for inst in instance]
        _results = [i.filtered_resps for i in instance]
        ctx = instance[0].args[0] if instance else ""
        return cls(doc=instance[0].doc, ctx=ctx, targets=targets, results=_results)

    def to_metric_inputs(self):
        return {"references": self.targets, "predictions": self.results}
