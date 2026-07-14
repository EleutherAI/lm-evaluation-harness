from dataclasses import dataclass
from typing import Literal


@dataclass
class ExtractionConfig:
    method: Literal["logprobs", "first_token", "regex", "json_path", "choice_match"] = (
        "logprobs"
    )
    metric: str | dict | None = None
    pattern: str | None = None  # For regex
    group: int = 1  # Regex capture group
    path: str | None = None  # For json_path
    normalize: str | None = None  # "lowercase", "strip", etc.
    fallback: str | None = None  # If extraction fails

    @classmethod
    def from_str(cls, method: str) -> "ExtractionConfig":
        return cls(method=method)

    def create_filters(self) -> list[dict] | None:
        if self.method == "logprobs":
            return None  # Handled elsewhere
        if self.method == "first_token":
            return [
                {
                    "name": "strict_match",
                    "filter": [{"function": "remove_whitespace"}],
                }
            ]
        return None

    def create_metrics(self) -> list[dict] | None:
        if self.method == "first_token":
            return [
                {
                    "metric": "exact_match",
                    "aggregation": "mean",
                    "higher_is_better": True,
                    "kwargs": {"ignore_case": True, "ignore_punctuation": True},
                }
            ]
        return None
