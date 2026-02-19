from __future__ import annotations

from typing import Any, Literal

from typing_extensions import Protocol


class ChatTemplate(Protocol):
    """Protocol for applying chat templates."""

    def __call__(
        self,
        chat_history: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        **kwargs,
    ) -> str | list[dict[str, Any]]: ...


# multiple_choice types send a number of "loglikelihood" instances
OutputType = Literal["loglikelihood", "loglikelihood_rolling", "generate_until"]
