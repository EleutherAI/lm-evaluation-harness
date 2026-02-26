from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import override

from lm_eval.api.instance import Instance

from ._task import Task


if TYPE_CHECKING:
    from collections.abc import Sequence

    from lm_eval.api._types import ChatTemplate
    from lm_eval.api.instance import GenInstance


class GenerateTask(Task):
    OUTPUT_TYPE = "generate_until"

    @override
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
    ) -> list[GenInstance]:
        metadata = metadata or {}

        target = self.doc_to_target(doc)
        arguments = (
            cast("str | list[dict[str, str]]", ctx),
            deepcopy(self.config.generation_kwargs),
        )

        multimodal_arguments = (
            self._build_multimodal_kwargs(doc) if self._multimodal else None
        )

        if self._multiple_targets:
            metadata.setdefault("metric_kwargs", {})["multiple_targets"] = True

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
