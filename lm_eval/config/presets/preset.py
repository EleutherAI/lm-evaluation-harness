from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal


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

    # Scorer
    scorer: str | dict[str, Any] | None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses that define preset_name
        if cls.preset_name is not None:
            PresetConfig._registry[cls.preset_name] = cls

    @classmethod
    def get(
        cls,
        spec: str | dict | PresetConfig | None,
        *,
        selection: str | None = None,
    ) -> PresetConfig | None:
        """Resolve preset from string, dict, or instance.

        Args:
            spec: One of:
                - str: Preset name ("mcqa", "cot", "generate")
                - dict with "type" key: single preset with overrides
                  ``{"type": "mcqa", "instruction": "..."}``
                - dict keyed by preset names: multi-preset lookup table
                  ``{"mcqa": {"instruction": "..."}, "generate": {...}}``
                - PresetConfig: Already instantiated
                - None: No preset
            selection: Which preset to pick from a multi-preset dict.
                Comes from ``task@selection`` syntax. If None, uses the
                first key as default.

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
                    f"Unknown preset: {spec}. Available: {cls.list_presets()}"
                )
            return template_cls()  # type: ignore

        if isinstance(spec, dict):
            # Single preset with overrides: {"type": "mcqa", "instruction": "..."}
            if "type" in spec:
                spec_copy = spec.copy()
                template_type = spec_copy.pop("type")
                template_cls = cls._registry.get(template_type)
                if template_cls is None:
                    raise ValueError(
                        f"Unknown preset type: {template_type}. "
                        f"Available: {cls.list_presets()}"
                    )
                return template_cls(**spec_copy)

            # Multi-preset lookup: {"mcqa": {...}, "generate": {...}}
            # Keys are preset names, values are override dicts (or None)
            if selection is not None:
                if selection not in spec:
                    raise ValueError(
                        f"Preset '{selection}' not found in task presets. "
                        f"Available: {list(spec.keys())}"
                    )
                chosen_name = selection
            else:
                # Default to first key
                chosen_name = next(iter(spec))

            overrides = spec[chosen_name]
            if overrides is None:
                overrides = {}
            elif isinstance(overrides, str):
                # Allow shorthand: {"mcqa": "some_other_preset"}
                return cls.get(overrides)
            return cls.get({"type": chosen_name, **overrides})

        raise TypeError(
            f"Invalid preset spec type: {type(spec)}. "
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

    @staticmethod
    def _escape_jinja(s: str) -> str:
        """Escape a string for inclusion in a Jinja template.

        Jinja templates are raw text, so quotes and backslashes are literal.
        Only Jinja delimiters ({{ }}, {% %}) would need escaping, but we
        don't expect those in preset string fields.
        """
        return s

    def to_jinja_config(
        self,
        doc_to_text: str = "question",
        doc_to_choice: str | None = "choices",
        doc_to_target: str = "answer",
    ) -> dict[str, str | None]:
        """Generate Jinja templates for TaskConfig fields.

        The doc_to_* parameters are field mappings from the task config that
        tell the preset which document fields to reference in templates.

        Returns a dict with:
            - doc_to_text: Jinja template for the prompt
            - doc_to_target: Jinja template for the target
            - doc_to_choice: Jinja template for choices (if applicable, else None)
        """
        return {
            "doc_to_text": self._build_doc_to_text_jinja(doc_to_text, doc_to_choice),
            "doc_to_target": self._build_doc_to_target_jinja(
                doc_to_target, doc_to_choice
            ),
            "doc_to_choice": self._build_doc_to_choice_jinja(doc_to_choice),
        }

    def _build_doc_to_text_jinja(
        self, doc_to_text: str, doc_to_choice: str | None
    ) -> str:
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
        template += self._field_ref(doc_to_text)

        # Choices (if applicable)
        if self.choice_labels and doc_to_choice:
            template += self._escape_jinja(self.before_choices)
            template += self._build_choices_format_jinja(doc_to_choice)

        # Answer section
        template += self._escape_jinja(self.before_answer)
        if self.answer_instruction:
            template += self._escape_jinja(
                self.answer_instruction + self.answer_instruction_delimiter
            )
        template += self._escape_jinja(self.answer_prompt)

        return template

    def _build_choices_format_jinja(self, doc_to_choice: str) -> str:
        r"""Build Jinja for formatting choices with labels.

        Generates Jinja like:
        {% for choice in choices %}{{ 'ABCD'[loop.index0] }}. {{ choice }}{% if not loop.last %}\n{% endif %}{% endfor %}
        """
        # Use _field_ref with for_output=False to strip {{ }} if present
        c_ref = self._field_ref(doc_to_choice, for_output=False)
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

    def _build_doc_to_target_jinja(
        self, doc_to_target: str, doc_to_choice: str | None
    ) -> str:
        """Build Jinja template for doc_to_target.

        For multiple_choice output_type, returns the index.
        For generate_until, returns the formatted answer text.
        """
        # Get field references - raw for control flow, wrapped for output
        a_raw = self._field_ref(doc_to_target, for_output=False)
        a_out = self._field_ref(doc_to_target, for_output=True)
        c_ref = (
            self._field_ref(doc_to_choice, for_output=False) if doc_to_choice else None
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

    def _build_doc_to_choice_jinja(self, doc_to_choice: str | None) -> str | None:
        """Build Jinja template for doc_to_choice.

        Returns choice labels if configured, otherwise the raw choices.
        """
        if not doc_to_choice:
            return None

        # Strip {{ }} if present to get raw field reference
        c_ref = self._field_ref(doc_to_choice, for_output=False)

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

    def to_task_config(
        self,
        doc_to_text: str = "question",
        doc_to_choice: str | None = "choices",
        doc_to_target: str = "answer",
    ) -> dict[str, Any]:
        """Expand preset into TaskConfig field overrides.

        The doc_to_* parameters are field mappings from the task config.
        The preset consumes them to build Jinja templates, then returns
        overrides that are applied unconditionally to the TaskConfig.

        Returns a dict of TaskConfig-compatible fields including:
        - Jinja templates (doc_to_text, doc_to_target, doc_to_choice)
        - Formatting fields (output_type, target_delimiter, etc.)
        - Scorer config (filter_list, metric_list) from extraction
        """
        cfg: dict[str, Any] = self.to_jinja_config(
            doc_to_text=doc_to_text,
            doc_to_choice=doc_to_choice,
            doc_to_target=doc_to_target,
        )

        # Formatting fields
        cfg["output_type"] = self.output_type
        cfg["target_delimiter"] = self.target_delimiter
        cfg["fewshot_delimiter"] = self.fewshot_delimiter
        if self.gen_prefix is not None:
            cfg["gen_prefix"] = self.gen_prefix

        # Scorer type — resolved by build_scorer() via the scorer registry
        if self.scorer is not None:
            cfg["scorer"] = self.scorer

        return cfg
