from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

from lm_eval.api.instance import GenInstance, Instance

from ._task import Task


if TYPE_CHECKING:
    from lm_eval.api._types import ChatTemplate, GenKwargs


class GenerateTask(Task):
    OUTPUT_TYPE = "generate_until"

    def construct_requests(
        self,
        doc: dict[str, Any],
        ctx: str | list[str] | list[dict[str, str]],
        *,
        doc_id: int,
        metadata: dict[str, Any] | None = None,
        apply_chat_template: bool = False,
        chat_template: ChatTemplate | None = None,
        **kwargs,
    ) -> list[GenInstance]:
        metadata = metadata or {}
        arguments = (
            cast("str | list[dict[str, str]]", ctx),
            deepcopy(self.config.generation_kwargs),
        )
        multimodal_arguments = (
            self._build_multimodal_kwargs(doc) if self._multimodal else None
        )
        target = self.doc_to_target(doc)
        if self._multiple_targets:
            metadata.setdefault("metric_ctx", {})["multiple_targets"] = True

        return [
            Instance(
                request_type=cast("Literal['generate_until']", self.OUTPUT_TYPE),
                doc=doc,
                arguments=arguments,
                additional_args=multimodal_arguments,
                target=target,
                idx=0,
                task_name=self.task_name,
                doc_id=doc_id,
                repeats=self.config.repeats,
                metadata={**metadata},
                **kwargs,
            )
        ]

    def _build_message_args(
        self,
        ctx: list[dict[str, str]],
    ) -> tuple[list[dict[str, str]], GenKwargs]:
        """Build arguments for generation when ctx is a raw message list.

        The model implementation receives the messages directly and handles
        formatting and generation.
        """
        return ctx, deepcopy(self.config.generation_kwargs)
