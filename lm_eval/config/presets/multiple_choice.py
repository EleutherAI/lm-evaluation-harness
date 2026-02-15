from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from lm_eval.config.presets.preset import PresetConfig


if TYPE_CHECKING:
    from lm_eval.config.presets.extraction import ExtractionConfig


@dataclass(kw_only=True)
class MCQPreset(PresetConfig):
    preset_name: ClassVar[str | None] = "mcq"

    # Mode
    output_type: str = "multiple_choice"

    instruction: str | None = None
    instruction_delimiter: str = ""

    # Question
    question_prefix: str | None = "Question:"
    prefix_delimiter: str = " "

    # Choices
    choice_labels: str | list[str] | None = "letters"
    choice_delimiter: str = "\n"
    before_choices: str = "\n"

    # Answer
    before_answer: str = "\n"
    answer_instruction: str | None = None  # CoT instruction
    answer_instruction_delimiter: str = ""  # After answer_instruction
    answer_prompt: str = "Answer:"
    answer_format: str = "letters"
    gen_prefix: str | None = None

    # Fewshot
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"

    # Extraction
    extraction: ExtractionConfig | str | None = None


@dataclass(kw_only=True)
class ClozePreset(MCQPreset):
    preset_name: ClassVar[str | None] = "cloze"

    choice_labels: str | list[str] | None = None
    answer_format: str = "full_text"
