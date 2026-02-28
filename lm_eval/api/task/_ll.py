from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

from lm_eval.api.instance import Instance
from lm_eval.api.utils import ends_with_whitespace
from lm_eval.config.templates import _coerce_list, _resolve_target_index, process_field

from ._task import Task


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lm_eval.api._types import ChatTemplate, Doc
    from lm_eval.api.instance import LLInstance
    from lm_eval.config import TaskConfig


eval_logger = logging.getLogger(__name__)


class MultipleChoiceTask(Task):
    OUTPUT_TYPE = "multiple_choice"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.repeats and self.config.repeats > 1:
            eval_logger.warning(
                "MultipleChoiceTask does not support repeats > 1, but config has repeats=%s. Setting repeats to 1.",
                self.config.repeats,
            )
        self.config.repeats = 1

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | Sequence[str] | list[dict[str, str]],
        *,
        doc_id: int,
        metadata: dict[str, Any] | None = None,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[LLInstance] | None:
        _metadata = {**(metadata or {})}

        choices = self.doc_to_choice(doc)
        target = self.doc_to_target(doc)
        if not choices or target is None:
            eval_logger.warning(
                "No choices=%s or target=%s found for doc:\n\n%s\n\nSkipping this instance.",
                choices,
                target,
                doc,
            )
            return None
        target_delimiter = self.config.target_delimiter
        if apply_chat_template:
            target_delimiter = (
                self.config.target_delimiter
                if self.config.gen_prefix
                and not ends_with_whitespace(self.config.gen_prefix)
                else ""
            )

        if isinstance(ctx, list) and isinstance(ctx[0], dict):
            # Message list context: delegate entirely to _build_chat_arguments
            return self._build_chat_arguments(
                doc=doc,
                ctx=cast("list[dict[str, str]]", ctx),
                choices=choices,
                target_delimiter=target_delimiter,
                target=target,
                doc_id=doc_id,
                metadata=_metadata,
                **kwargs,
            )

        # From here on, ctx should always be a str (or list[str] for multiple_inputs)
        if self._multiple_inputs:
            arguments = self._multiple_input_args(
                context=cast("list[str]", ctx),
                choices=choices,
                target_delimiter=target_delimiter,
            )
        else:
            arguments = [
                (cast("str", ctx), f"{target_delimiter}{cont}") for cont in choices
            ]

        # If any scorer uses acc_mutual_info, we need unconditional loglikelihoods.
        # This computes log(P(choice|ctx) / P(choice)) = log(P(choice|ctx)) - log(P(choice))
        # by appending ("", continuation) pairs for each choice.
        # NOTE: this will at most ~2x runtime.
        arg_meta = [(arg, {**_metadata}) for arg in arguments]
        if self._has_metric("acc_mutual_info"):
            aux_arguments = self.build_mutual_info(
                context="", choices=choices, target_delimiter=target_delimiter
            )
            arg_meta.extend(
                (arg, {**_metadata, "acc_mutual_info": True}) for arg in aux_arguments
            )

        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                task_name=self.task_name,
                idx=i,
                doc_id=doc_id,
                repeats=self.config.repeats,
                target=target,
                metadata=meta,
                **kwargs,
            )
            for i, (arg, meta) in enumerate(arg_meta)
        ]

    @staticmethod
    def build_mutual_info(
        *, context="", choices: list[str], target_delimiter: str
    ) -> list[tuple[str, str]]:
        assert choices is not None and target_delimiter is not None, (
            "choices and target_delimiter must be provided to create acc_mutual_info auxiliary arguments"
        )
        return [(context, f"{target_delimiter}{choice}") for choice in choices]

    def _build_chat_arguments(
        self,
        doc: dict[str, Any],
        ctx: list[dict[str, str]],
        choices: Sequence[str],
        target_delimiter: str,
        target: Any,
        *,
        doc_id: int,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[LLInstance] | None:
        """Build Instance list when ctx is a raw message list.

        The model implementation handles loglikelihood extraction directly
        from the message format. The continuation is always a plain string.
        """
        if self._multiple_inputs:
            raise NotImplementedError

        arguments = [
            (deepcopy(ctx), f"{target_delimiter}{choice}") for choice in choices
        ]

        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,  # type:ignore[invalid-argument-type]
                task_name=self.task_name,
                idx=i,
                doc_id=doc_id,
                repeats=self.config.repeats,
                target=target,
                metadata=metadata,
                **kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

    @staticmethod
    def _multiple_input_args(
        *, context: Sequence[str], choices: Sequence[str], target_delimiter: str
    ) -> list[tuple[str, str]]:
        assert isinstance(context, list) and isinstance(context[0], str), (
            "For multiple input tasks, ctx should be a list of strings"
        )
        assert len(choices) == 1, (
            "For multiple input tasks, there should only be one choice"
        )
        return [(cxt, f"{target_delimiter}{choices[0]}") for cxt in context]

    def doc_to_text(
        self,
        doc: Doc,
        doc_to_text: Callable[[Doc], str | list[str]] | str | None = None,
    ) -> str | list[str] | None:
        doc_to_text = (
            doc_to_text if doc_to_text is not None else self.config.doc_to_text
        )
        y = process_field(doc, doc_to_text)
        if self._multiple_inputs:
            y = _coerce_list(y)
        return y

    def doc_to_choice(
        self,
        doc: Doc,
        doc_to_choice: Callable[[Doc], list[str]] | str | list[str] | None = None,
    ) -> list[str] | None:
        choices = super().doc_to_choice(doc, doc_to_choice)
        if (
            choices is not None
            and not isinstance(choices, list)
            and not isinstance(choices[0], str)
        ):
            eval_logger.warning(
                "doc_to_choice should return a list of strings representing the answer choices. Skipping ..."
            )
            return None
        if self._multiple_inputs:
            assert choices is not None and len(choices) == 1, (
                "For multiple input tasks, doc_to_choice should return a list with a single string representing the answer choice template."
            )
        return choices

    def doc_to_target(
        self,
        doc: Doc,
        doc_to_target: Callable[[Doc], str | int | list[int] | list[str]]
        | str
        | None = None,
    ) -> int | list[int] | None:
        target = super().doc_to_target(doc, doc_to_target)
        choices = self.doc_to_choice(doc)
        if not choices:
            eval_logger.warning(
                "No choices found for doc:\n\n%s\n\nCannot map non int target to choice index.",
                doc,
            )
            return None

        if isinstance(target, list):
            acc = [
                idx
                for t in target
                if (idx := _resolve_target_index(t, choices, doc)) is not None
            ]
            if not acc:
                eval_logger.warning("No valid targets found in doc_to_target list.")
                return None
            return acc

        return _resolve_target_index(target, choices, doc)

    def set_repeats(self, repeats: int) -> None:
        """Override the default number of repeats this task."""
        eval_logger.debug(
            "[%s] Ignoring attempt to set repeats, as MultipleChoiceTask does not support repeats > 1.",
            self.task_name,
        )


class LoglikelihoodTask(Task):
    OUTPUT_TYPE: Literal["loglikelihood"] = "loglikelihood"

    def __init__(self, config: TaskConfig | dict[str, Any]):
        super().__init__(config)
        assert self._multiple_inputs is False
        assert self._multiple_targets is False
        if self.config.repeats and self.config.repeats > 1:
            eval_logger.warning(
                "LoglikelihoodTask does not support repeats > 1, but config has repeats=%s. Setting repeats to 1.",
                self.config.repeats,
            )
        self.config.repeats = 1

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | Sequence[str] | list[dict[str, Any]],
        *,
        doc_id: int,
        metadata: dict[str, Any] | None = None,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[LLInstance]:
        metadata = {**(metadata or {})}
        cont = self.doc_to_target(doc)
        assert isinstance(ctx, str), (
            f"For loglikelihood tasks, the argument should be a string representing the continuation to score against the context. Got type {type(ctx)} with value {ctx}. Please check your doc_to_text implementation."
        )
        assert isinstance(cont, str), (
            f"For loglikelihood tasks, the target should be a string representing the continuation to score. Got {cont} of type {type(cont)}. Please check your doc_to_target implementation."
        )
        arguments = (ctx, cont)
        if self._multiple_targets:
            metadata.setdefault("metric_kwargs", {})["multiple_targets"] = True

        return [
            Instance(
                request_type=cast("Literal['loglikelihood']", self.OUTPUT_TYPE),
                doc=doc,
                arguments=arguments,
                task_name=self.task_name,
                idx=0,
                doc_id=doc_id,
                repeats=self.config.repeats,
                target=0,
                metadata=metadata,
                **kwargs,
            )
        ]

    def doc_to_target(
        self,
        doc: Doc,
        doc_to_target: Callable[[Doc], str | int | list[int] | list[str]]
        | str
        | None = None,
    ) -> str | None:
        target = super().doc_to_target(doc, doc_to_target)
        if not isinstance(target, str):
            eval_logger.warning(
                "doc_to_target should return a string representing the continuation to score for LoglikelihoodTask. Got %s of type %s. Skipping this instance.",
                target,
                type(target),
            )
            return None
        return target

    def set_repeats(self, repeats: int) -> None:
        """Override the default number of repeats this task."""
        eval_logger.debug(
            "[%s] Ignoring attempt to set repeats, as LoglikelihoodTask does not support repeats > 1.",
            self.task_name,
        )


class LoglikelihoodRollingTask(LoglikelihoodTask):
    OUTPUT_TYPE = "loglikelihood_rolling"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | Sequence[str] | list[dict[str, Any]],
        *,
        doc_id: int,
        metadata: dict[str, Any] | None = None,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[LLInstance]:
        # Rolling loglikelihood scores the full target text with no context prefix.
        # Convention: args = (context, continuation) matching loglikelihood —
        # context is empty for rolling, continuation is the text being scored.
        text = self.doc_to_target(doc)
        assert isinstance(text, str), (
            f"doc_to_target must return a string for loglikelihood_rolling tasks. Got {type(text)}."
        )
        arguments = cast("tuple[str, str]", ("", text))

        return [
            Instance(
                request_type=cast("Literal['loglikelihood_rolling']", self.OUTPUT_TYPE),
                doc=doc,
                arguments=arguments,
                task_name=self.task_name,
                idx=0,
                doc_id=doc_id,
                repeats=self.config.repeats,
                target=0,
                metadata={**(metadata or {})},
                **kwargs,
            )
        ]
