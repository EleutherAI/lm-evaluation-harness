from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal
from typing_extensions import TypedDict


OutputType = Literal[
    "loglikelihood", "loglikelihood_rolling", "generate_until", "multiple_choice"
]


class _DOCTO(TypedDict, extra_items=str):
    doc_to_text: str
    doc_to_target: str
    doc_to_choice: str | None


_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _label_at_index(choice_labels: str | list[str] | None, index_expr: str) -> str:
    """Return a Jinja expression for the label at a 0-based index.

    Args:
        choice_labels: ``"letters"``, ``"numbers"``, a custom list, or ``None``.
        index_expr: Jinja expression evaluating to a 0-based integer
            (e.g. ``"loop.index0"``, ``"answer"``).

    Examples:
        >>> _label_at_index("letters", "loop.index0")
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[loop.index0]"
        >>> _label_at_index("numbers", "answer")
        "answer + 1"
        >>> _label_at_index(["I", "II", "III"], "loop.index0")
        "['I', 'II', 'III'][loop.index0]"
    """
    if choice_labels == "letters":
        return f"'{_LETTERS}'[{index_expr}]"
    elif choice_labels == "numbers":
        return f"{index_expr} + 1"
    elif isinstance(choice_labels, list):
        return f"{choice_labels!s}[{index_expr}]"
    return "''"


def _labels_up_to(choice_labels: str | list[str] | None, count_expr: str) -> str:
    """Return a Jinja expression for the list of labels up to *count_expr*.

    Args:
        choice_labels: ``"letters"``, ``"numbers"``, a custom list, or ``None``.
        count_expr: Jinja expression evaluating to an integer count
            (e.g. ``"_num_choices"``, ``"choices|length"``).

    Examples:
        >>> _labels_up_to("letters", "_num_choices")
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:_num_choices]|list"
        >>> _labels_up_to("numbers", "choices|length")
        "range(1, choices|length + 1)|list"
        >>> _labels_up_to(["I", "II", "III"], "_num_choices")
        "['I', 'II', 'III'][:_num_choices]"
    """
    if choice_labels == "letters":
        return f"'{_LETTERS}'[:{count_expr}]|list"
    elif choice_labels == "numbers":
        return f"range(1, {count_expr} + 1)|list"
    elif isinstance(choice_labels, list):
        return f"{choice_labels!s}[:{count_expr}]"
    return "''"


@dataclass(kw_only=True)
class FormatConfig:
    """Declarative prompt configuration for evaluation tasks.

    A format defines **what** a prompt looks like (instruction, question,
    choices, answer) and **how** sections are separated, then compiles
    itself into Jinja templates consumed by ``TaskConfig``.

    Prompt layout produced by ``to_jinja_config()``:

        ```text
        {instruction}
        {question_prefix}{question}
        {section_separator}{choice_label}. {choice}  (repeated)
        {section_separator}{answer_instruction}{answer_prompt}
        {target_delimiter}{target}
        {fewshot_delimiter}
        ... next example ...
        ```

    Subclasses register themselves by setting ``format_name`` and are
    looked up via ``FormatConfig.get("name")`` or the YAML
    ``format: name`` / ``task@name`` syntax.

    Derived Jinja variables
    ~~~~~~~~~~~~~~~~~~~~~~~
    When ``choice_labels`` and ``doc_to_choice`` are set, a preamble of
    ``{% set %}`` blocks is emitted at the top of the template, making
    the following variables available inside ``instruction``,
    ``answer_prompt``, and other fields that accept ``{{ }}`` refs:

    - ``_num_choices`` — number of choices (int), e.g. ``4``
    - ``_choice_labels`` — label list, e.g. ``['A', 'B', 'C', 'D']``
    - ``_choice_list_and`` — natural-language with "and",
      e.g. ``"A, B, C and D"``
    - ``_choice_list_or`` — natural-language with "or",
      e.g. ``"A, B, C or D"``
    """

    # Registry for format subclasses
    _registry: ClassVar[dict[str, type[FormatConfig]]] = {}
    format_name: ClassVar[str | None] = None
    """Set in subclasses to auto-register (e.g. ``format_name = "mcqa"``)."""

    output_type: OutputType
    """Model request type: ``"multiple_choice"`` for loglikelihood scoring,
    ``"generate_until"`` for free-form generation, etc."""

    instruction: str | None
    r"""Task instruction prepended to every prompt. Include any trailing
    delimiter (e.g. ``"Choose the best answer.\n"``). ``None`` for no
    instruction."""

    question_prefix: str | None
    """Label before the question text. Include trailing whitespace
    (e.g. ``"Question: "``, ``"Problem: "``). ``None`` for no prefix."""

    choice_labels: str | list[str] | None = None
    """How to label answer choices. ``"letters"`` for A/B/C/D,
    ``"numbers"`` for 1/2/3/4, a custom list like ``["I", "II", "III"]``,
    or ``None`` to show choices without labels."""

    choice_delimiter: str
    r"""Separator between individual choice items (typically ``"\n"``)."""

    section_separator: str
    r"""Separator inserted between major prompt sections
    (question → choices, choices → answer). Typically ``"\n"``."""

    answer_instruction: str | None = None
    r"""Optional instruction before the answer prompt, used for
    chain-of-thought (e.g. ``"Think step by step.\n"``). Include any
    trailing delimiter. ``None`` to omit."""

    answer_prompt: str
    """Text soliciting the answer (e.g. ``"Answer:"``,
    ``'Your response should end with "The answer is [X]".'``)."""

    gen_prefix: str | None
    """Constrained-decoding prefix appended to the prompt so the model
    continues from a known anchor (e.g. ``"The best answer is"``).
    ``None`` for unconstrained generation."""

    target_delimiter: str
    r"""Separator between the prompt and the target value in few-shot
    examples (e.g. ``" "`` or ``"\n"``)."""

    fewshot_delimiter: str
    r"""Separator between few-shot examples (typically ``"\n\n"``)."""

    scorer: str | dict[str, Any] | None
    """Scoring method name or config. Resolved by the scorer registry
    (e.g. ``"first_token"``, ``None`` for default)."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-register subclasses that define format_name
        if cls.format_name is not None:
            FormatConfig._registry[cls.format_name] = cls

    @property
    def answer_format(self) -> str:
        """How the target/ground-truth is formatted for ``generate_until`` tasks.

        ``"letters"`` converts an index to ``A``/``B``/``C``,
        ``"numbers"`` to ``1``/``2``/``3``, ``"full_text"`` to the choice text.
        Ignored for ``multiple_choice`` output type.
        """
        # Todo: handle custom list of labels if self.choice_labels is a list
        if self.choice_labels == "letters":
            return "letters"
        elif self.choice_labels == "numbers":
            return "numbers"
        return "full_text"

    @classmethod
    def get(
        cls,
        spec: str | dict | FormatConfig | None,
        *,
        selection: str | None = None,
    ) -> FormatConfig | None:
        """Resolve format from string, dict, or instance.

        Args:
            spec: One of:
                - str: Format name ("mcqa", "cot", "generate")
                - dict with "type" key: single format with overrides
                  ``{"type": "mcqa", "instruction": "..."}``
                - dict keyed by format names: multi-format lookup table
                  ``{"mcqa": {"instruction": "..."}, "generate": {...}}``
                - FormatConfig: Already instantiated
                - None: No format
            selection: Which format to pick from a multi-format dict.
                Comes from ``task@selection`` syntax. If None, uses the
                first key as default.

        Returns:
            FormatConfig instance or None
        """
        if spec is None:
            return None
        if isinstance(spec, FormatConfig):
            return spec

        if isinstance(spec, str):
            template_cls = cls._registry.get(spec)
            if template_cls is None:
                raise ValueError(
                    f"Unknown format: {spec}. Available: {cls.registered()}"
                )
            return template_cls()  # type: ignore

        if isinstance(spec, dict):
            # Single format with overrides: {"type": "mcqa", "instruction": "..."}
            if "type" in spec:
                spec_copy = spec.copy()
                template_type = spec_copy.pop("type")
                template_cls = cls._registry.get(template_type)
                if template_cls is None:
                    raise ValueError(
                        f"Unknown format type: {template_type}. "
                        f"Available: {cls.registered()}"
                    )
                return template_cls(**spec_copy)

            # Multi-format lookup: {"mcqa": {...}, "generate": {...}}
            # Keys are format names, values are override dicts (or None)
            if selection is not None:
                if selection not in spec:
                    raise ValueError(
                        f"Format '{selection}' not found in task formats. "
                        f"Available: {list(spec.keys())}"
                    )
                chosen_name = selection
            else:
                # Default to the first key
                chosen_name = next(iter(spec))

            overrides = spec[chosen_name]
            if overrides is None:
                overrides = {}
            elif isinstance(overrides, str):
                # Allow shorthand: {"mcqa": "some_other_format"}
                return cls.get(overrides)
            return cls.get({"type": chosen_name, **overrides})

        raise TypeError(
            f"Invalid format spec type: {type(spec)}. "
            f"Expected str, dict, FormatConfig, or None."
        )

    @classmethod
    def registered(cls) -> list[str]:
        """List all registered format names."""
        return sorted(cls._registry.keys())

    def _field_ref(self, field: str, for_output: bool = True) -> str:
        """Convert the field name to Jinja reference.

        Args:
            field: Field name or Jinja expression
            for_output: If True, wrap in {{ }} for output. If False, return raw for use in control flow.

        If a field contains, '{{' it's already a Jinja expression.
        """
        if "{{" in field:
            # Already a Jinja expression - extract the inner part for control flow
            if for_output:
                return field.strip()
            else:
                # Strip {{ }} for use in {% if %} etc.
                inner = field.replace("{{", "").replace("}}", "").strip()
                # Parenthesize complex expressions so |filter, .method(),
                # and [idx] bind to the whole expression, not just the last
                # token. E.g. [a, b, c]|length → ([a, b, c])|length
                if not inner.isidentifier():
                    return "(" + inner + ")"
                return inner
        if for_output:
            return "{{" + field + "}}"
        return field

    @staticmethod
    def _escape_jinja(s: str) -> str:
        """No-op currently — returns the string unchanged.

        Marks every site where a format string field (``instruction``,
        ``answer_prompt``, etc.) is spliced into a Jinja template. If
        format fields ever need to contain literal ``{{``/``{%``
        delimiters, replace this with real escaping (e.g. wrapping in
        ``{% raw %}...{% endraw %}``).
        """
        return s

    def to_jinja_config(
        self,
        doc_to_text: str = "question",
        doc_to_choice: str | list | None = "choices",
        doc_to_target: str | int = "answer",
    ) -> _DOCTO:
        r"""Generate Jinja templates for TaskConfig fields.

        The doc_to_* parameters are field mappings from the task config that
        tell the format which document fields to reference in templates.

        Args:
            doc_to_text: Field name or Jinja expression for the question text.
                Common forms found in task YAML configs:

                - Plain field name — ``"question"``
                - Jinja field ref — ``"{{question}}"`` (idempotent with above)
                - Nested field — ``"{{question.stem}}"``
                - Bracket access — ``"{{row['question']}}"``
                - With filter — ``"{{ question.strip() }}"``
                - Multi-field — ``"{{context}}\\n{{question}}"``
                - Join expression — ``"{{[s1, s2, s3]|join(' ')}}"``
                - String slicing — ``"{{text.split(' ')[:-1]|join(' ')}}"``

            doc_to_choice: Field name, Jinja expression, Python list, or None.
                Common forms found in task YAML configs:

                - Plain field name — ``"choices"``
                - Nested field — ``"{{choices.text}}"``
                - List construction — ``"{{[answerA, answerB, answerC]}}"``
                - Map/filter — ``"{{answers|map(attribute='atext')|list}}"``
                - Hardcoded list — ``["yes", "no", "maybe"]``
                  (arrives as a Python ``list``; converted internally)
                - ``None`` — no choices (valid for generation tasks)

            doc_to_target: Field name, Jinja expression, or integer constant
                for the answer. Common forms found in task YAML configs:

                - Plain field name — ``"answer"``, ``"label"``, ``"gold"``
                - Jinja field ref — ``"{{label}}"`` (idempotent with above)
                - Integer constant — ``0``, ``3``
                - Index lookup — ``"{{choices.label.index(answerKey)}}"``
                - Arithmetic — ``"{{ (label|int) - 1 }}"``
                - Conditional — ``"{{choice1 if label == 0 else choice2}}"``
                - Array mapping — ``"{{['B', 'A'][label]}}"``
                - String ops — ``"{{answer.split(' ')[0]}}"``
                - With filter — ``"{{answer_number|string}}"``

        Returns:
            A dict with keys:

            - doc_to_text: Jinja template for the prompt.
            - doc_to_target: Jinja template for the target.
            - doc_to_choice: Jinja template for choices, or None.
        """
        # Integer doc_to_target (e.g. YAML `doc_to_target: 0`) is a constant,
        # not a field name. Convert to a Jinja literal.
        if isinstance(doc_to_target, int):
            doc_to_target = "{{" + str(doc_to_target) + "}}"
        if isinstance(doc_to_text, int):
            doc_to_text = "{{" + str(doc_to_text) + "}}"

        # Hardcoded YAML lists (e.g. doc_to_choice: ["yes", "no"]) arrive as
        # Python lists.  Convert to a string literal that is valid Jinja syntax.
        if isinstance(doc_to_choice, list):
            doc_to_choice = str(doc_to_choice)

        return {
            "doc_to_text": self._build_doc_to_text_jinja(doc_to_text, doc_to_choice),
            "doc_to_target": self._build_doc_to_target_jinja(
                doc_to_target, doc_to_choice
            ),
            "doc_to_choice": self._build_doc_to_choice_jinja(doc_to_choice),
        }

    def _build_preamble_jinja(self, doc_to_choice: str | None) -> str:
        r"""Emit ``{% set %}`` blocks that compute choice-derived variables.

        These variables are available to ``instruction``, ``answer_prompt``,
        and other fields that may contain ``{{ }}`` references:

        - ``_num_choices`` — integer count of choices
        - ``_choice_labels`` — list of label strings (e.g. ``['A', 'B', 'C', 'D']``)
        - ``_choice_list_and`` — natural-language enumeration with "and"
          (e.g. ``"A, B, C and D"``)
        - ``_choice_list_or`` — same but with "or"
          (e.g. ``"A, B, C or D"``)

        Returns an empty string when ``doc_to_choice`` is None or
        ``choice_labels`` is not set.
        """
        if not doc_to_choice or not self.choice_labels:
            return ""

        c_ref = self._field_ref(doc_to_choice, for_output=False)
        preamble = "{% set _num_choices = " + c_ref + "|length %}"

        preamble += (
            "{% set _choice_labels = "
            + _labels_up_to(self.choice_labels, "_num_choices")
            + " %}"
        )

        preamble += (
            "{% set _choice_list_and = "
            "_choice_labels[:-1]|join(', ') ~ ' and ' ~ _choice_labels[-1]|string %}"
        )
        preamble += (
            "{% set _choice_list_or = "
            "_choice_labels[:-1]|join(', ') ~ ' or ' ~ _choice_labels[-1]|string %}"
        )

        return preamble

    def _build_doc_to_text_jinja(
        self, doc_to_text: str, doc_to_choice: str | None
    ) -> str:
        """Build Jinja template for doc_to_text.

        Generates a template that produces:
        {preamble}
        {instruction}
        {question_prefix}{question}
        {section_separator}{formatted_choices}
        {section_separator}{answer_instruction}{answer_prompt}
        """
        preamble = self._build_preamble_jinja(doc_to_choice)
        template = ""

        # Named section separators — derived from section_separator for now
        # but kept as local names so they can be split out later if needed.
        before_choices = self.section_separator
        before_answer = self.section_separator

        # Instruction (may contain Jinja refs like {{ _num_choices }})
        if self.instruction:
            template += self.instruction

        # Question prefix and question
        if self.question_prefix:
            template += self._escape_jinja(self.question_prefix)
        template += self._field_ref(doc_to_text)

        # Choices (if applicable)
        if self.choice_labels and doc_to_choice:
            template += self._escape_jinja(before_choices)
            template += self._build_choices_format_jinja(doc_to_choice)

        # Answer section
        template += self._escape_jinja(before_answer)
        if self.answer_instruction:
            template += self.answer_instruction
        # answer_prompt may contain Jinja refs like {{ _choice_list_or }}
        template += self.answer_prompt

        # Only prepend the preamble if the body actually references
        # any of the computed variables; skip it to keep templates clean.
        _PREAMBLE_VARS = (
            "_num_choices",
            "_choice_labels",
            "_choice_list_and",
            "_choice_list_or",
        )
        if preamble and any(v in template for v in _PREAMBLE_VARS):
            template = preamble + template

        return template

    def _build_choices_format_jinja(self, doc_to_choice: str) -> str:
        r"""Build Jinja for formatting choices with labels.

        Generates a ``{% for %}`` loop that pairs each choice with a label.

        Args:
            doc_to_choice: Field name or Jinja expression resolving to an
                iterable of choice strings (e.g. ``"choices"``,
                ``"{{[answerA, answerB, answerC]}}"``,
                ``"{{answers|map(attribute='atext')|list}"``).

        Returns:
            A Jinja template string. For example, with
            ``choice_labels="letters"`` and ``doc_to_choice="choices"``:

                ```jinja
                {% for choice in choices %}
                    {{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[loop.index0] }}. {{ choice }}
                {% endfor %}
                ```

            Which renders (given ``choices = ["Paris", "London", "Berlin"]``)
            as:

                ```text
                A. Paris
                B. London
                C. Berlin
                ```

            Label style is controlled by ``self.choice_labels``:

            - ``"letters"`` — ``A. choice``, ``B. choice``, ...
            - ``"numbers"`` — ``1. choice``, ``2. choice``, ...
            - ``["I", "II", ...]`` — custom list indexed by position
            - ``None`` — no label prefix (renders ``. choice``)
        """
        # Use _field_ref with for_output=False to strip {{ }} if present
        c_ref = self._field_ref(doc_to_choice, for_output=False)
        delim = self._escape_jinja(self.choice_delimiter)

        labels_expr = _label_at_index(self.choice_labels, "loop.index0")

        return (
            "{% for choice in " + c_ref + " %}"
            "{{ " + labels_expr + " }}. {{ choice }}"
            "{% if not loop.last %}" + delim + "{% endif %}"
            "{% endfor %}"
        )

    def _build_doc_to_target_jinja(
        self, doc_to_target: str, doc_to_choice: str | None
    ) -> str:
        """Build a Jinja template for doc_to_target.

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
            if c_ref is None:
                raise ValueError(
                    f"Format '{self.format_name}' has output_type='multiple_choice' "
                    f"but no doc_to_choice was provided. Set doc_to_choice in your "
                    f"task config (e.g. doc_to_choice: choices) so the format can "
                    f"build the target template."
                )
            return (
                "{% if " + a_raw + " is number %}" + a_out + "{% else %}"
                "{{ " + c_ref + ".index(" + a_raw + ") }}"
                "{% endif %}"
            )
        elif self.output_type == "generate_until":
            # For generate_until, return a formatted answer based on answer_format
            if self.answer_format in ("letters", "numbers"):
                # Convert index to label (A/B/C or 1/2/3)
                return (
                    "{% if " + a_raw + " is number %}"
                    "{{ " + _label_at_index(self.choice_labels, a_raw) + " }}"
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
        """Build a Jinja template for doc_to_choice.

        Returns choice labels if configured, otherwise the raw choices.
        """
        if not doc_to_choice:
            return None

        # Strip {{ }} if present to get raw field reference
        c_ref = self._field_ref(doc_to_choice, for_output=False)

        if not self.choice_labels:
            # Return raw choices
            return "{{ " + c_ref + " }}"

        # Return labels based on the number of choices
        return "{{ " + _labels_up_to(self.choice_labels, c_ref + "|length") + " }}"

    def to_task_config(
        self,
        doc_to_text: str = "question",
        doc_to_choice: str | list[str] | None = "choices",
        doc_to_target: str = "answer",
    ) -> dict[str, Any]:
        """Expand format into TaskConfig field overrides.

        The doc_to_* parameters are field mappings from the task config.
        The format consumes them to build Jinja templates, then returns
        overrides that are applied unconditionally to the TaskConfig.

        Returns a dict of TaskConfig-compatible fields including
        - Jinja templates (doc_to_text, doc_to_target, doc_to_choice)
        - Formatting fields (output_type, target_delimiter, etc.)
        - Scorer config (filter_list, metric_list) from extraction
        """
        cfg = {
            **self.to_jinja_config(
                doc_to_text=doc_to_text,
                doc_to_choice=doc_to_choice,
                doc_to_target=doc_to_target,
            ),
            "output_type": self.output_type,
            "target_delimiter": self.target_delimiter,
            "fewshot_delimiter": self.fewshot_delimiter,
        }

        # Formatting fields
        if self.gen_prefix is not None:
            cfg["gen_prefix"] = self.gen_prefix

        # Scorer type — resolved by build_scorer() via the scorer registry
        if self.scorer is not None:
            cfg["scorer"] = self.scorer

        return cfg
