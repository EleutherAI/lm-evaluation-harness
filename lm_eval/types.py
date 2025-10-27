# ruff: noqa: F401
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)


if TYPE_CHECKING:
    import datasets

    from lm_eval.api.instance import GenInstance, Instance, MCInstance

InstanceT = TypeVar("InstanceT", bound="Instance", contravariant=True)
ResultsSelf = TypeVar("ResultsSelf", bound="Results[Any]")


# input formats used in tasks
TaskDataSet = Union["datasets.Dataset", Iterable[dict[str, Any]]]  # noqa: UP007
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


@runtime_checkable
class Results(Protocol[InstanceT]):
    """Class for storing results of a task."""

    results: Any
    target: Any
    scores: Any
    instances: Sequence[InstanceT]

    @classmethod
    def from_instances(
        cls: type[ResultsSelf],
        results: Sequence[InstanceT],
        filter_name: str = "default",
    ) -> ResultsSelf: ...

    @abstractmethod
    def to_metric_inputs(self) -> Any: ...

    @staticmethod
    def create(instance: Sequence[InstanceT], filter_name: str | None = None):
        output_type = instance[0].request_type
        match output_type:
            case "loglikelihood":
                return MCResult.from_instances(instance)
            case _:
                return GenResult.from_instances(instance, filter_name=filter_name)


@dataclass
class MCResult(Results):
    """Result of a multiple-choice task. Instances are grouped by doc_id beforehand"""

    lls: Sequence[float]
    is_greedy: Sequence[bool]
    target: int
    instances: Sequence["MCInstance"]
    choices: Sequence[str] = field(default_factory=list)
    char_lens: Sequence[int] = field(default_factory=list)
    byte_lens: Sequence[int] = field(default_factory=list)
    token_lens: Sequence[int] = field(default_factory=list)
    lls_mutual_info: Sequence[float] = field(default_factory=list)
    scores: dict[Any, float] = field(default_factory=dict)

    @classmethod
    def from_instances(cls, results: Sequence["MCInstance"], acc_mutual_info=False):
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
            instances=instance,
        )

    def to_metric_inputs(self):
        return self


@dataclass
class GenResult:
    """Result of a generation task, grouped by doc_id.

    Handles multiple generations per doc (e.g., temperature sampling, repeats).
    """

    results: Sequence[str]  # All generated texts
    target: str | list[str]  # Gold reference(s)
    instances: Sequence["GenInstance"]  # Source instances
    repeats: int = 1  # Number of repeats/samples per doc
    filter_name: str = "default"  # Active filter
    scores: dict[Any, list[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def is_repeated(self) -> bool:
        """Check if this result has multiple samples."""
        return self.repeats > 1

    @classmethod
    def from_instances(
        cls, instances: Sequence["GenInstance"], filter_name: str | None = None
    ) -> "GenResult":
        """Create GenResult from instances for the same doc_id.

        Args:
            instances: List of Instance objects for the same doc
            filter_name: Name of filter to use for filtered responses

        """
        if not instances:
            raise ValueError("Cannot create GenResult from empty instances")

        # All instances should have the same doc and target
        doc = instances[0].doc
        target = [x.target for x in instances]
        # Targets should generally be the same for all instances in a sample
        if all(x.target == target[0] for x in instances):
            target = target[0]

        # Extract generations from filtered responses
        generations = []
        for inst in instances:
            if filter_name and filter_name in inst.filtered_resps:
                resp = inst.filtered_resps[filter_name]
                # Handle both single and multiple responses
                if isinstance(resp, list):
                    generations.extend(resp)
                else:
                    generations.append(resp)
            else:
                # # Fallback to raw response if filter not found
                # if isinstance(inst.resps, list):
                #     generations.extend(inst.resps)
                # else:
                generations.append(inst.filtered_resps)

        return cls(
            results=generations,
            target=target,
            instances=instances,
            repeats=len(generations),
            filter_name=filter_name,
        )

    def to_metric_inputs(self):
        return {
            "predictions": self.results,
            "references": self.target,
        }
