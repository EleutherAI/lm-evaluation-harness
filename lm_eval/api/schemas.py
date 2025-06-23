from dataclasses import dataclass
from typing import Optional


# @dataclass
# class GenerateInput:
#     """
#     Inputs for the generate function.
#     """
#
#     prompt: str
#     gen_kwargs: dict
#     multimodal_arg: Optional[dict] = None
#
#     def __iter__(self):
#         return (
#             iter((self.prompt, self.gen_kwargs))
#             if not self.multimodal_arg
#             else iter((self.prompt, self.gen_kwargs, self.multimodal_arg))
#         )
#
#     def __getitem__(self, item: int):
#         return [self.prompt, self.gen_kwargs][item]
#
#
# @dataclass
# class GenerateOutput:
#     """
#     Outputs for the generate function.
#     """
#
#     text: str
#     metadata: dict = None
#
#
# @dataclass
# class LoglikelihoodInput:
#     """
#     Inputs for the loglikelihood function.
#     """
#
#     context: str
#     continuation: Optional[str] = None
#
#
# class LoglikelihoodOutput(NamedTuple):
#     """
#     Outputs for the loglikelihood function.
#     """
#
#     loglikelihood: float
#     is_greedy: Optional[bool] = None
#     ctx_tokens: Optional[list[int]] = None
#     cont_tokens: Optional[list[int]] = None
#     metadata: Optional[dict] = None

# def __iter__(self):
#     return iter((self.loglikelihood, self.is_greedy))


@dataclass
class MetricResult:
    """
    Outputs for the metric function.
    """

    doc_id: str | int | None
    scores: list[dict[str, float]] | dict
    filter_key: str = None
    metric_name: str = None
    metadata: Optional[dict] = None

    def __iter__(self):
        if self.scores is None:
            return iter([])

        # Group values by metric key
        if not isinstance(self.scores, list):
            self.scores = [self.scores]
        grouped = {}
        for score_dict in self.scores:
            for key, value in score_dict.items():
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(value)

        # Return iterator of (key, list[values]) pairs
        return iter(grouped.items())

    def get_metric_results(self, metric_key) -> list[float]:
        if self.scores is None:
            return []
        return [
            score_dict[metric_key]
            for score_dict in self.scores
            if metric_key in score_dict
        ]

    @property
    def metric_keys(self) -> list[str]:
        if self.scores is None:
            return []
        return (
            list(self.scores[0].keys())
            if isinstance(self.scores, list)
            else list(self.scores.keys())
        )
