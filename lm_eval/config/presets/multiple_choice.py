from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .preset import PresetConfig


@dataclass(kw_only=True)
class MCQPreset(PresetConfig):
    preset_name: ClassVar[str | None] = "mcqa"

    # Mode
    output_type: str = "multiple_choice"

    instruction: str | None = None

    # Question
    question_prefix: str | None = "Question: "

    # Choices
    choice_labels: str | list[str] | None = "letters"
    choice_delimiter: str = "\n"

    # Layout
    section_separator: str = "\n"

    # Answer
    answer_instruction: str | None = None  # CoT instruction
    answer_prompt: str = "Answer:"
    gen_prefix: str | None = None

    # Fewshot
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"

    # Scorer
    scorer: str | None = None


@dataclass(kw_only=True)
class ClozePreset(MCQPreset):
    preset_name: ClassVar[str | None] = "cloze"

    choice_labels: str | list[str] | None = None
