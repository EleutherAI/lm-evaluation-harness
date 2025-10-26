from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypedDict


if TYPE_CHECKING:
    import datasets

    from lm_eval.api.instance import MCInstance


# input formats used in tasks
TaskDataSet = datasets.Dataset | Iterable[dict[str, Any]]
DatasetSplits = dict[str, TaskDataSet]
ChatFormat = str | list[dict[str, Any]]

# output results for loglikelihood and generate_until methods
GenerateResult = list[str]
LLResult = tuple[float, bool | None]


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


class MultiModalArgs(TypedDict, total=False):
    image: dict[str, Any]
    audio: dict[str, Any]
    video: dict[str, Any]


class ChatTemplateProtocol(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs,
    ) -> ChatFormat: ...


@dataclass
class MCResult:
    """Result of a multiple-choice task. Instances are grouped by doc_id beforehand"""

    lls: Sequence[float]
    is_greedy: Sequence[bool]
    target: int
    choices: Sequence[str]
    char_lens: Sequence[int] = field(default_factory=list)
    byte_lens: Sequence[int] = field(default_factory=list)
    token_lens: Sequence[int] = field(default_factory=list)
    lls_mutual_info: Sequence[float] = field(default_factory=list)

    @classmethod
    def from_instances(cls, results: list["MCInstance"], acc_mutual_info=False):
        import numpy as np

        ## TODO: ADD Choice/Target Verification
        instance = sorted(
            results,
            key=lambda x: (x.doc_id, x.metadata.get("acc_mutual_info", False)),
        )
        resps, choices, targets = zip(
            *((inst.resps, inst.args[1], inst.target) for inst in instance), strict=True
        )
        lls, is_greedy = zip(*resps, strict=True)
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
                raise ValueError
            # And this stores our "regular" conditional loglikelihoods
            lls = lls[: len(choices)]
            lls_mutual_info = [
                ll_c - ll_u for ll_c, ll_u in zip(lls, lls_unconditional, strict=True)
            ]

        # calculate lengths
        completion_len = np.array([float(len(i)) for i in choices])
        bytes_len = np.array([len(i.encode("utf-8")) for i in choices])
        token_len = np.array(inst.token_len.get("cont", 1) for inst in instance)
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
            token_lens=token_len,  # type: ignore
            lls_mutual_info=lls_mutual_info,
        )
