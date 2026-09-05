"""Optional lm-evaluation-harness plugin for ThinkRetrieve.

This module is deliberately kept outside the package's runtime dependencies.
Install ``lm-eval`` and ``thinkretrieve[faiss]`` to use it::

    python -m lm_eval --model thinkretrieve \
      --model_args=model=qwen3:4b,base_url=http://localhost:11434/v1,bank_path=gsm8k_bank \
      --tasks gsm8k --include_path integrations/lm_eval

The harness passes one prompt at a time to ``generate_until``. The adapter
uses the prompt as the ThinkRetrieve question and returns the final answer.
It is intended for generative tasks (GSM8K-style), not log-likelihood tasks.
The bank directory must be a ThinkRetrieve ``FaissRetriever.save`` output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
except ImportError as exc:  # pragma: no cover - exercised only without lm-eval
    raise ImportError(
        "Install lm-eval to use the ThinkRetrieve harness plugin"
    ) from exc

from thinkretrieve import FaissRetriever, OpenAICompatBackend, ThinkRetrieve, ThinkRetrieveConfig


def _args(raw: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"model_args item needs key=value: {item!r}")
        values[key.strip()] = value.strip()
    return values


@register_model("thinkretrieve")
class ThinkRetrieveLM(LM):
    """A generation-only lm-evaluation-harness model wrapper."""

    REQUIRES_VLLM = False

    def __init__(
        self,
        model: str,
        bank_path: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "EMPTY",
        think_budget: int = 2048,
        max_insertions: int = 3,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        del device  # Kept for CLI compatibility with harness model arguments.
        self._model = model
        self._backend = OpenAICompatBackend(
            model=model, base_url=base_url, api_key=api_key
        )
        self._retriever = FaissRetriever.load(bank_path)
        self._config = ThinkRetrieveConfig(
            think_budget=int(think_budget), max_insertions=int(max_insertions)
        )

    @property
    def eot_token_id(self) -> Optional[int]:
        return None

    @property
    def max_length(self) -> int:
        return self._config.think_budget + self._config.answer_max_tokens

    @property
    def max_gen_toks(self) -> int:
        return self._config.answer_max_tokens

    @property
    def batch_size(self) -> int:
        return 1

    def tok_encode(self, string: str, **_: Any) -> List[int]:
        # The adapter is generation-only; harness token accounting is an
        # estimate because the remote backend owns tokenization.
        return list(range(self._backend.count_tokens(string)))

    def tok_decode(self, tokens: List[int], **_: Any) -> str:
        return " " * len(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Any]:
        raise NotImplementedError(
            "ThinkRetrieveLM supports generation tasks, not log-likelihood tasks"
        )

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[Any]:
        raise NotImplementedError(
            "ThinkRetrieveLM supports generation tasks, not rolling likelihoods"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        outputs: List[str] = []
        for request in requests:
            context = request.args[0]
            result = ThinkRetrieve(
                self._backend, self._retriever, self._config
            ).run(context)
            outputs.append(result.answer.strip())
        return outputs


def build_model(args: str) -> ThinkRetrieveLM:
    """Small programmatic entry point useful in notebooks and tests."""
    return ThinkRetrieveLM(**_args(args))
