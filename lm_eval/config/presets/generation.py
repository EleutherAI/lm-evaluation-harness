from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from lm_eval.config.presets.preset import PresetConfig


if TYPE_CHECKING:
    from lm_eval.config.presets.extraction import ExtractionConfig


@dataclass(kw_only=True)
class GeneratePreset(PresetConfig):
    preset_name: ClassVar[str | None] = "generate"

    # Mode
    output_type: str = "generate_until"

    instruction: str = "Given the following question and four candidate answers (A, B, C and D), choose the best answer."
    instruction_delimiter: str = "\n"

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
    answer_prompt: str = 'Your response should end with "The best answer is [the_answer_letter]" where the [the_answer_letter] is one of A, B, C or D.'
    answer_format: str = "letters"
    gen_prefix: str = "The best answer is"

    # Fewshot
    target_delimiter: str = "\n"
    fewshot_delimiter: str = "\n\n"

    # Extraction
    extraction: ExtractionConfig | str | None = "first_token"


@dataclass(kw_only=True)
class COTGeneratePreset(GeneratePreset):
    preset_name: ClassVar[str | None] = "cot"

    instruction: str = (
        "Given the following problem, reason step by step to find the final answer."
    )
    question_prefix: str = "Problem:"
    choice_labels: None = None
    answer_format: str = "full_text"
    answer_prompt: str = 'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    gen_prefix: None = None
