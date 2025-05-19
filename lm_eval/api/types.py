from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerateInput:
    """
    Inputs for the generate function.
    """

    prompt: str
    gen_kwargs: dict
    multimodal_arg: Optional[dict] = None


@dataclass
class GenerateOutput:
    """
    Outputs for the generate function.
    """

    text: str
    metadata: dict = None


@dataclass
class LoglikelihoodInput:
    """
    Inputs for the loglikelihood function.
    """

    context: str
    continuation: Optional[str] = None


@dataclass
class LoglikelihoodOutput:
    """
    Outputs for the loglikelihood function.
    """

    loglikelihood: float
    is_greedy: Optional[bool] = None
    ctx_tokens: Optional[list[int]] = None
    cont_tokens: Optional[list[int]] = None
    metadata: Optional[dict] = None

    def __iter__(self):
        return iter((self.loglikelihood, self.is_greedy))
