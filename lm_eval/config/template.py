from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable


if TYPE_CHECKING:
    from lm_eval.config.metric import MetricConfig


@dataclass
class TemplateConfig:
    """Encapsulates information about a template."""

    template: str
    doc_to_text: str | Callable[[dict], str]
    doc_to_choice: str | list | Callable[[dict], list]
    doc_to_target: int | Callable[[dict], int]
    description: str
    context_prefix: str
    prefix_delimiter: str
    context_delimiter: str
    answer_suffix: str
    target_delimiter: str
    choice_format: str | None
    choice_delimiter: str | None
    fewshot_delimiter: str
    metric_list: list[str] | list[MetricConfig] | None = field(
        default_factory=lambda: ["acc", "acc_norm"]
    )


@dataclass
class MCQTemplateConfig(TemplateConfig):
    """Encapsulates information about a template.
    Would return a sample with the following format:
    Question: <doc_to_text(doc)>
    A. <doc_to_choice(doc)[0]>
    B. <doc_to_choice(doc)[1]>
    C. <doc_to_choice(doc)[2]>
    D. <doc_to_choice(doc)[3]>
    Answer:` doc_to_choice(doc)` for each choice.
    """

    doc_to_text: str | Callable[[dict], str]
    doc_to_choice: str | list | Callable[[dict], list]
    doc_to_target: int | Callable[[dict], int]
    template = "mcq"
    context_prefix: str = "Question:"
    prefix_delimiter: str = " "
    context_delimiter: str = "\n"
    answer_suffix: str = "Answer:"
    target_delimiter: str = "\n"
    choice_format: str | None = "letters"
    choice_delimiter: str | None = "\n"
    fewshot_delimiter: str = "\n\n"
    metric_list: list[MetricConfig] | None = field(default_factory=lambda: ["acc"])


@dataclass
class ClozeTemplateConfig(TemplateConfig):
    """Encapsulates information about a template.
    Would return a sample with the following format:
    Question:  <doc_to_text(doc)>
    Answer:` <doc_to_target(doc)>`
    """

    doc_to_text: str | Callable[[dict], str]
    doc_to_choice: str | list | Callable[[dict], list]
    doc_to_target: int | Callable[[dict], int]
    template: str = "cloze"
    description: str = ""
    context_prefix: str = "Question:"
    prefix_delimiter: str = " "
    context_delimiter: str = "\n"
    answer_suffix: str = "Answer:"
    target_delimiter: str = " "
    choice_format: str | None = None
    choice_delimiter: str | None = None
    fewshot_delimiter: str = "\n\n"
    metric_list: list[MetricConfig] | None = field(
        default_factory=lambda: ["acc", "acc_norm"]
    )
