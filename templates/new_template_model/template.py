"""
Skeleton for a new local LM implementation.

Copy this file to ``lm_eval/models/<your_model_filename>.py`` and fill in the
TODOs. See ``docs/model_guide.md`` for the full walkthrough and ``HFLM`` in
``lm_eval/models/huggingface.py`` for a fully-featured reference.

The two main building blocks used here are:

- ``TemplateLM`` (``lm_eval.api.model.TemplateLM``): provides shared
  tokenization / scoring boilerplate so subclasses only need to implement
  ``_loglikelihood_tokens`` and ``generate_until``. Subclass this directly
  for new local models that don't fit the HuggingFace ``transformers`` flow.
- ``Collator`` (``lm_eval.models.utils.Collator``): reorders requests by
  length (descending) and yields batches, so padding stays minimal and the
  longest example is processed first (catches OOMs early).

This skeleton intentionally omits a full ``loglikelihood_tokens`` implementation
because that part is highly model-specific. The TODOs mark the only places you
must touch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lm_eval.api.model import TemplateLM
from lm_eval.models.utils import Collator


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


# IMPORTANT: do not name this module the same as a pypi package you import
# (e.g. ``anthropic.py`` would shadow the ``anthropic`` SDK). Prefer
# ``<vendor>_llms.py`` or ``<framework>_lm.py``.

eval_logger = logging.getLogger(__name__)


# TODO: uncomment and replace ``my-model`` with the CLI name(s) you want to
# expose, e.g. ``@register_model("my-model", "my-model-chat")``. Keep this
# decorator commented out in the template so importing it doesn't pollute the
# real model registry.
#
# from lm_eval.api.registry import register_model
# @register_model("my-model")
class MyTemplateLM(TemplateLM):
    """Replace this docstring with a one-liner describing your model backend.

    Construction kwargs are forwarded from ``--model_args``. Keep the signature
    close to ``HFLM.__init__`` if you want to share documentation.
    """

    # Maximum sequence length used when the model config doesn't expose one.
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        batch_size: int | str = 1,
        max_length: int | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        # TODO: load your model + tokenizer here. Store everything you need
        #       on ``self`` so the request methods can reuse it.
        super().__init__()
        self._pretrained = pretrained
        self._batch_size = batch_size
        self._max_length = max_length or self._DEFAULT_MAX_LENGTH
        self._device = device

    # ------------------------------------------------------------------
    # Properties expected by the harness
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        # TODO: return the model's end-of-text token id (often
        # ``self.tokenizer.eos_token_id``). Used by the evaluator for stop logic.
        raise NotImplementedError

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        # Default cap for ``generate_until`` when a task does not set
        # ``max_gen_toks``. 256 is a reasonable starting value.
        return 256

    @property
    def batch_size(self) -> int:
        return (
            int(self._batch_size)
            if isinstance(self._batch_size, str)
            else self._batch_size
        )

    @property
    def device(self):
        return self._device

    # ------------------------------------------------------------------
    # Tokenization (only required if subclassing TemplateLM)
    # ------------------------------------------------------------------

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        # TODO: tokenize ``string`` to a list of ids. Whatever you return here
        #       is what loglikelihood scoring will operate on.
        raise NotImplementedError

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        # TODO: inverse of ``tok_encode``. Used to render generated outputs.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Required request methods
    # ------------------------------------------------------------------

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Score each ``(context, continuation)`` pair.

        Returns a list of ``(logprob, is_greedy)`` tuples in the same order
        as ``requests``.
        """
        # TODO: implement, or subclass ``TemplateLM`` and implement
        #       ``_loglikelihood_tokens`` instead. The base class will then
        #       handle prompt construction, batching and ordering for you.
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute full-sequence log-likelihood for perplexity tasks.

        The base ``TemplateLM`` provides a sliding-window implementation built
        on ``loglikelihood``; override only if you can do something faster.
        """
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Greedy / sampled generation up to per-request stop strings.

        Each ``Instance.args`` is ``(context: str, gen_kwargs: dict)``, where
        ``gen_kwargs`` may include ``until``, ``max_gen_toks``, ``temperature``,
        ``do_sample``, etc.
        """

        # ------- 1) reorder + batch with Collator -------
        # Sort by descending input length so padding stays minimal and the
        # longest example is generated first (catches OOM early).
        def _sort_key(req: tuple[str, dict]) -> int:
            ctx, _ = req
            return -len(self.tok_encode(ctx))

        # ``group_by="gen_kwargs"`` keeps requests with identical generation
        # parameters in the same batch — important because most backends can't
        # mix ``until`` strings or ``max_gen_toks`` within a single forward pass.
        collator = Collator(
            [req.args for req in requests],
            sort_fn=_sort_key,
            group_by="gen_kwargs",
        )

        results: list[str] = []
        for batch in collator.get_batched(n=self.batch_size):
            # ``batch`` is a list of ``(context, gen_kwargs)`` tuples that share
            # the same gen_kwargs.
            contexts = [ctx for ctx, _ in batch]
            gen_kwargs = batch[0][1] if batch else {}

            # TODO: call your model here. Translate ``gen_kwargs`` into the
            #       generate() arguments expected by your backend (e.g. map
            #       ``until`` -> stop sequences, ``max_gen_toks`` -> max_new_tokens).
            outputs: list[str] = self._generate(contexts, gen_kwargs)

            # ------- 2) optional partial caching -------
            # ``cache_hook.add_partial`` writes the request/response pair into
            # the SQLite cache (when ``--use_cache`` was set). This lets later
            # runs short-circuit identical requests without re-hitting the model.
            for (ctx, gk), out in zip(batch, outputs, strict=True):
                self.cache_hook.add_partial("generate_until", (ctx, gk), out)
                results.append(out)

        # ------- 3) restore original order -------
        # The Collator hands batches back in length-sorted order; the harness
        # expects responses indexed the same as the input requests.
        return collator.get_original(results)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate(self, contexts: list[str], gen_kwargs: dict) -> list[str]:
        # TODO: implement model-specific generation here.
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional: chat-template support
    # ------------------------------------------------------------------
    # Implement these to make ``--apply_chat_template``, ``--system_instruction``
    # and ``--fewshot_as_multiturn`` work with your model. See
    # ``HFLM.apply_chat_template`` for a reference.

    @property
    def tokenizer_name(self) -> str:
        # Used to fingerprint cached requests across chat templates.
        raise NotImplementedError

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        # Return the Jinja chat template string (or "" for no template).
        return ""

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        raise NotImplementedError
