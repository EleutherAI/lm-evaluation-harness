"""Cross-platform ONNX Runtime GenAI backend for lm-eval-harness.

Runs models exported by the ONNX Runtime GenAI *Model Builder* through the
``onnxruntime-genai`` runtime (``og.Model`` / ``og.Generator``). Unlike the
``winml`` backend -- which selects execution providers via the Windows ML
catalog and therefore only runs on Windows -- this backend selects providers
with the cross-platform ``og.Config`` API and runs on CPU, CUDA, DirectML,
WebGPU, and AMD NPU (VitisAI/RyzenAI).

Example usage:
    lm_eval --model onnxruntime-genai --model_args pretrained=path/to/model_builder_output,execution_provider=cuda --tasks hellaswag --limit 10

The ``pretrained`` path is a Model Builder output directory (containing
``genai_config.json``, the ONNX graph(s), and an HF tokenizer) or a ``.onnx``
file inside such a directory.

All lm-eval logic lives in :class:`lm_eval.models._onnx_base._ONNXLMBase`,
which is shared with the raw ``onnxruntime`` backend; this module implements
only the GenAI engine primitives.
"""

import logging
from typing import Any

import numpy as np

from lm_eval.api.registry import register_model
from lm_eval.models._onnx_base import _log_softmax, _ONNXLMBase


eval_logger = logging.getLogger(__name__)

# Re-exported for backwards compatibility: the scoring helper now lives in the
# shared base module.
__all__ = ["ONNXRuntimeGenAILM", "_log_softmax"]

# ONNX Runtime GenAI treats CPU as "no execution provider appended" rather than
# an appendable provider name, so these aliases are handled specially.
_CPU_EP_ALIASES = {"cpu", "cpuexecutionprovider"}


@register_model("onnxruntime-genai")
class ONNXRuntimeGenAILM(_ONNXLMBase):
    """Cross-platform backend for Model Builder ONNX models via onnxruntime-genai.

    Subclasses need only override :meth:`_select_ep` to change how execution
    providers are chosen (see :class:`lm_eval.models.winml.WindowsML`).
    """

    @staticmethod
    def _import_genai():
        try:
            import onnxruntime_genai as og
        except ImportError as e:
            raise ImportError(
                "onnxruntime-genai is required for the onnxruntime-genai backend. "
                "Install a matching EP wheel, e.g. `pip install onnxruntime-genai` "
                "(CPU) or `pip install onnxruntime-genai-cuda` (CUDA)."
            ) from e
        eval_logger.info(f"ONNX Runtime GenAI version: {og.__version__}")
        return og

    # ------------------------------------------------------------------ #
    # Engine hooks
    # ------------------------------------------------------------------ #
    def _select_ep(self, config) -> None:
        """Configure execution providers on an ``og.Config``.

        ``og.Config(model_dir)`` seeds its provider list from
        ``genai_config.json``, so we only touch it when the user explicitly
        asks for a provider:

        * ``execution_provider is None`` — leave the config untouched, honoring
          the providers the export was built with (a CUDA/DML/NPU export runs
          on its target device without extra flags).
        * ``execution_provider`` is a CPU alias — clear providers so it runs on
          CPU (an empty provider list means CPU).
        * otherwise — clear and append the named provider with any
          ``provider_options``.
        """
        ep = self.execution_provider
        if ep is None:
            eval_logger.info(
                "No execution_provider given; using the providers declared in "
                "genai_config.json."
            )
            return
        config.clear_providers()
        if ep.lower() in _CPU_EP_ALIASES:
            eval_logger.info("Using CPU execution provider (no provider appended).")
            return
        config.append_provider(ep)
        for key, value in self.provider_options.items():
            config.set_provider_option(ep, str(key), str(value))
        eval_logger.info(f"Using execution provider: {ep}")

    def _load(self) -> None:
        """Load the onnxruntime-genai model on the requested provider."""
        self.og = self._import_genai()
        config = self.og.Config(str(self.model_dir))
        self._select_ep(config)
        self.model = self.og.Model(config)
        eval_logger.info(f"Loaded onnxruntime-genai model from {self.model_dir}")

    def _forward_logits(self, tokens: list[int]) -> np.ndarray:
        params = self.og.GeneratorParams(self.model)
        # Without this, the search max_length defaults to the model's full
        # context_length from genai_config.json, so every request allocates a
        # KV cache for that whole span (34x slower on Qwen1.5-0.5B, identical
        # logits). Only the prompt pass is needed here, so pin it to the input.
        params.set_search_options(max_length=len(tokens), do_sample=False)
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(np.asarray(tokens, dtype=np.int32))
        logits = np.asarray(generator.get_output("logits"), dtype=np.float32)
        if logits.ndim == 3:  # (batch, seq, vocab)
            logits = logits[0]
        elif logits.ndim != 2:
            raise ValueError(f"Unexpected logits shape from GenAI: {logits.shape}")
        return logits

    def _generate(self, tokens: list[int], gen_kwargs: dict[str, Any]) -> list[int]:
        max_gen_toks = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))
        temperature = float(gen_kwargs.get("temperature", 0.0))
        top_p = float(gen_kwargs.get("top_p", 1.0))
        top_k = int(gen_kwargs.get("top_k", 50))
        do_sample = bool(gen_kwargs.get("do_sample", False)) and temperature > 0

        params = self.og.GeneratorParams(self.model)
        # max_length is total (prompt + generated) length for the genai search.
        total_max = min(len(tokens) + max_gen_toks, self.max_length)
        if do_sample:
            params.set_search_options(
                max_length=total_max,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            params.set_search_options(max_length=total_max, do_sample=False)

        generator = self.og.Generator(self.model, params)
        generator.append_tokens(np.asarray(tokens, dtype=np.int32))

        generated: list[int] = []
        while not generator.is_done() and len(generated) < max_gen_toks:
            generator.generate_next_token()
            seq = generator.get_sequence(0)
            if len(seq) > len(tokens) + len(generated):
                generated.append(int(seq[len(tokens) + len(generated)]))
        return generated
