from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .preset import PresetConfig


@dataclass(kw_only=True)
class GeneratePreset(PresetConfig):
    preset_name: ClassVar[str | None] = "generate"

    # Mode
    output_type: str = "generate_until"

    instruction: str = "Given the following question and {{ _num_choices }} candidate answers ({{ _choice_list_and }}), choose the best answer.\n"

    # Question
    question_prefix: str | None = "Question: "

    # Choices
    choice_labels: str | list[str] | None = "letters"
    choice_delimiter: str = "\n"

    # Layout
    section_separator: str = "\n"

    # Answer
    answer_instruction: str | None = None  # CoT instruction
    answer_prompt: str = 'Your response should end with "The best answer is [answer_letter]" where the [answer_letter] is one of {{ _choice_list_or }}.'
    gen_prefix: str = "The best answer is"

    # Fewshot
    target_delimiter: str = "\n"
    fewshot_delimiter: str = "\n\n"

    # Scorer
    scorer: str | None = "first_token"


@dataclass(kw_only=True)
class COTGeneratePreset(GeneratePreset):
    preset_name: ClassVar[str | None] = "cot"

    instruction: str = (
        "Given the following problem, reason step by step to find the final answer.\n"
    )
    question_prefix: str = "Problem: "
    choice_labels: None = None
    answer_prompt: str = 'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    gen_prefix: None = None
