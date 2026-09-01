from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal


if TYPE_CHECKING:
    from lm_eval.config.presets.extraction import ExtractionConfig


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


@dataclass(kw_only=True)
class PresetConfig:
    # Registry for preset subclasses
    _registry: ClassVar[dict[str, type[PresetConfig]]] = {}
    preset_name: ClassVar[str | None] = None  # Set in subclasses to auto-register

    # Mode
    output_type: OutputType

    instruction: str | None
    instruction_delimiter: str  # After instruction

    # Question
    question_prefix: str | None  # "Question: ", "Problem: "
    prefix_delimiter: str

    # Choices
    choice_labels: str | list[str] | None = None
    choice_delimiter: str  # Between each choice
    before_choices: str  # Between question and first choice

    # Answer
    before_answer: str  # Between choices/question and answer section
    answer_instruction: str | None = None  # CoT instruction
    answer_instruction_delimiter: str  # After answer_instruction
    answer_prompt: str  # "Answer:", "The answer is", etc.
    answer_format: str
    gen_prefix: str | None

    # Fewshot
    target_delimiter: str  # Between answer_prompt and target
    fewshot_delimiter: str  # Between examples

    # Extraction
    extraction: ExtractionConfig | str | None

    # Field mappings - map preset fields to document fields
    # These can be simple field names or Jinja expressions
    doc_to_text: str = "question"
    doc_to_choice: str | None = "choices"
    doc_to_target: str = "answer"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses that define preset_name
        if cls.preset_name is not None:
            PresetConfig._registry[cls.preset_name] = cls

    def __post_init__(self):
        from lm_eval.config.presets.extraction import ExtractionConfig

        if isinstance(self.extraction, str):
            self.extraction = ExtractionConfig.from_str(self.extraction)

    @classmethod
    def get(cls, spec: str | dict | PresetConfig | None) -> PresetConfig | None:
        """Resolve template from string, dict, or instance.

        Args:
            spec: One of:
                - str: Template name ("mcq", "cot", "generate")
                - dict: {"type": "mcq", "instruction": "..."}
                - PresetConfig: Already instantiated
                - None: No template

        Returns:
            PresetConfig instance or None
        """
        if spec is None:
            return None
        if isinstance(spec, PresetConfig):
            return spec

        if isinstance(spec, str):
            template_cls = cls._registry.get(spec)
            if template_cls is None:
                raise ValueError(
                    f"Unknown template: {spec}. Available: {cls.list_presets()}"
                )
            return template_cls()  # type: ignore

        if isinstance(spec, dict):
            spec_copy = spec.copy()
            template_type = spec_copy.pop("type", "default")
            template_cls = cls._registry.get(template_type)
            if template_cls is None:
                raise ValueError(
                    f"Unknown template type: {template_type}. "
                    f"Available: {cls.list_presets()}"
                )
            return template_cls(**spec_copy)

        raise TypeError(
            f"Invalid template spec type: {type(spec)}. "
            f"Expected str, dict, PresetConfig, or None."
        )

    @classmethod
    def list_presets(cls) -> list[str]:
        """List all registered preset names."""
        return sorted(cls._registry.keys())

    def _field_ref(self, field: str, for_output: bool = True) -> str:
        """Convert field name to Jinja reference.

        Args:
            field: Field name or Jinja expression
            for_output: If True, wrap in {{ }} for output. If False, return raw for use in control flow.

        If field contains '{{' it's already a Jinja expression.
        """
        if "{{" in field:
            # Already a Jinja expression - extract the inner part for control flow
            if for_output:
                return field.strip()
            else:
                # Strip {{ }} for use in {% if %} etc.
                return field.replace("{{", "").replace("}}", "").strip()
        if for_output:
            return "{{" + field + "}}"
        return field

    def _escape_jinja(self, s: str) -> str:
        """Escape a string for inclusion in Jinja template.

        Preserves actual newlines as-is since Jinja handles them correctly.
        """
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def to_jinja_config(self) -> dict[str, str | None]:
        """Generate Jinja templates for TaskConfig fields.

        Returns a dict with:
            - doc_to_text: Jinja template for the prompt
            - doc_to_target: Jinja template for the target
            - doc_to_choice: Jinja template for choices (if applicable, else None)
        """
        return {
            "doc_to_text": self._build_doc_to_text_jinja(),
            "doc_to_target": self._build_doc_to_target_jinja(),
            "doc_to_choice": self._build_doc_to_choice_jinja(),
        }

    def _build_doc_to_text_jinja(self) -> str:
        """Build Jinja template for doc_to_text.

        Generates a template that produces:
        {instruction}{instruction_delimiter}
        {question_prefix}{prefix_delimiter}{question}
        {before_choices}{formatted_choices}
        {before_answer}{answer_instruction}{answer_instruction_delimiter}{answer_prompt}
        """
        template = ""

        # Instruction
        if self.instruction:
            template += self._escape_jinja(
                self.instruction + self.instruction_delimiter
            )

        # Question prefix and question
        if self.question_prefix:
            template += self._escape_jinja(self.question_prefix + self.prefix_delimiter)
        template += self._field_ref(self.doc_to_text)

        # Choices (if applicable)
        if self.choice_labels and self.doc_to_choice:
            template += self._escape_jinja(self.before_choices)
            template += self._build_choices_format_jinja()

        # Answer section
        template += self._escape_jinja(self.before_answer)
        if self.answer_instruction:
            template += self._escape_jinja(
                self.answer_instruction + self.answer_instruction_delimiter
            )
        template += self._escape_jinja(self.answer_prompt)

        return template

    def _build_choices_format_jinja(self) -> str:
        r"""Build Jinja for formatting choices with labels.

        Generates Jinja like:
        {% for choice in choices %}{{ 'ABCD'[loop.index0] }}. {{ choice }}{% if not loop.last %}\n{% endif %}{% endfor %}
        """
        assert self.doc_to_choice is not None, (
            "choices_field required for choice formatting"
        )
        # Use _field_ref with for_output=False to strip {{ }} if present
        c_ref = self._field_ref(self.doc_to_choice, for_output=False)
        delim = self._escape_jinja(self.choice_delimiter)

        if self.choice_labels == "letters":
            labels_expr = "'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[loop.index0]"
        elif self.choice_labels == "numbers":
            labels_expr = "loop.index"
        elif isinstance(self.choice_labels, list):
            labels_list = str(self.choice_labels)
            labels_expr = f"{labels_list}[loop.index0]"
        else:
            labels_expr = "''"

        return (
            "{% for choice in " + c_ref + " %}"
            "{{ " + labels_expr + " }}. {{ choice }}"
            "{% if not loop.last %}" + delim + "{% endif %}"
            "{% endfor %}"
        )

    def _build_doc_to_target_jinja(self) -> str:
        """Build Jinja template for doc_to_target.

        For multiple_choice output_type, returns the index.
        For generate_until, returns the formatted answer text.
        """
        # Get field references - raw for control flow, wrapped for output
        a_raw = self._field_ref(self.doc_to_target, for_output=False)
        a_out = self._field_ref(self.doc_to_target, for_output=True)
        c_ref = (
            self._field_ref(self.doc_to_choice, for_output=False)
            if self.doc_to_choice
            else None
        )

        if self.output_type == "multiple_choice":
            # For multiple choice, we need the index
            # If answer is already an index, return it; if it's text, find it in choices
            assert c_ref is not None, "choices_field required for multiple_choice"
            return (
                "{% if " + a_raw + " is number %}" + a_out + "{% else %}"
                "{{ " + c_ref + ".index(" + a_raw + ") }}"
                "{% endif %}"
            )
        elif self.output_type == "generate_until":
            # For generate_until, return formatted answer based on answer_format
            if self.answer_format == "letters":
                # Convert index to letter label (A, B, C, D)
                return (
                    "{% if " + a_raw + " is number %}"
                    "{{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[" + a_raw + "] }}"
                    "{% else %}" + a_out + "{% endif %}"
                )
            elif self.answer_format == "numbers":
                # Convert index to 1-based number (1, 2, 3, 4)
                return (
                    "{% if " + a_raw + " is number %}"
                    "{{ " + a_raw + " + 1 }}"
                    "{% else %}" + a_out + "{% endif %}"
                )
            elif self.answer_format == "full_text" and c_ref:
                # Return the full text of the choice
                return (
                    "{% if " + a_raw + " is number %}"
                    "{{ " + c_ref + "[" + a_raw + "] }}"
                    "{% else %}" + a_out + "{% endif %}"
                )

        # Default: return answer as-is
        return a_out

    def _build_doc_to_choice_jinja(self) -> str | None:
        """Build Jinja template for doc_to_choice.

        Returns choice labels if configured, otherwise the raw choices.
        """
        if not self.doc_to_choice:
            return None

        # Strip {{ }} if present to get raw field reference
        c_ref = self._field_ref(self.doc_to_choice, for_output=False)

        if not self.choice_labels:
            # Return raw choices
            return "{{ " + c_ref + " }}"

        # Return labels based on number of choices
        if self.choice_labels == "letters":
            return "{{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:" + c_ref + "|length] | list }}"
        elif self.choice_labels == "numbers":
            return "{{ range(1, " + c_ref + "|length + 1) | list }}"
        elif isinstance(self.choice_labels, list):
            labels_str = str(self.choice_labels)
            return "{{ " + labels_str + "[:" + c_ref + "|length] }}"

        return "{{ " + c_ref + " }}"

    def get_metrics(self):
        """Return metrics this template requires (or None for defaults)."""
        if self.extraction:
            return self.extraction.create_metrics()
        return None

    def get_filters(self):
        """Return filters this template requires (or None for defaults)."""
        if self.extraction:
            return self.extraction.create_filters()
        return None
