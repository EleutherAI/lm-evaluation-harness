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
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model


if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)

# ONNX Runtime GenAI treats CPU as "no execution provider appended" rather than
# an appendable provider name, so these aliases are handled specially.
_CPU_EP_ALIASES = {"cpu", "cpuexecutionprovider"}


@register_model("onnxruntime-genai")
class ONNXRuntimeGenAILM(TemplateLM):
    """Cross-platform backend for Model Builder ONNX models via onnxruntime-genai.

    All lm-eval scoring logic lives here (tokenization, single-pass logits,
    log-likelihood, rolling perplexity, and generation). Subclasses need only
    override :meth:`_select_ep` to change how execution providers are chosen
    (see :class:`lm_eval.models.winml.WindowsML`).
    """

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        pretrained: str,
        execution_provider: str | None = None,
        max_length: int | None = None,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        provider_options: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the backend.

        Args:
            pretrained: Path to a Model Builder output directory or a ``.onnx``
                file inside one.
            execution_provider: EP to run on (e.g. ``cpu``, ``cuda``, ``dml``,
                ``VitisAI``). When ``None`` (default), the providers declared in
                ``genai_config.json`` are used as-is — matching how a Model
                Builder export is meant to run on its target device. Pass
                ``cpu`` to force CPU, or a provider name to override the export.
            max_length: Maximum sequence length. Defaults to the
                ``context_length`` from ``genai_config.json`` when available.
            batch_size: Only 1 is supported; other values are coerced to 1.
            max_gen_toks: Default maximum number of tokens to generate.
            provider_options: Extra key/value options forwarded to the EP.
        """
        super().__init__()

        self.og = self._import_genai()
        self.pretrained = pretrained
        self.model_dir = self._resolve_model_dir(pretrained)
        self.execution_provider = execution_provider
        self.provider_options = provider_options or {}
        self._max_gen_toks = max_gen_toks

        if batch_size != 1:
            eval_logger.warning(
                f"{type(self).__name__} currently supports batch size 1 only; "
                f"overriding requested batch_size={batch_size}."
            )
        self.batch_size = 1

        self._genai_config = self._read_genai_config(self.model_dir)
        model_cfg = self._genai_config.get("model", {})
        context_length = model_cfg.get("context_length")
        self.max_length = max_length or context_length or self._DEFAULT_MAX_LENGTH

        self._eot_token_id = self._resolve_eot_token_id(model_cfg)
        self._bos_token_id = model_cfg.get("bos_token_id")

        self._load()

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

    @staticmethod
    def _resolve_model_dir(pretrained: str) -> Path:
        path = Path(pretrained)
        if path.is_file() and path.suffix == ".onnx":
            return path.parent
        if path.is_dir():
            return path
        raise FileNotFoundError(
            f"Model path {pretrained} not found or is not a directory / .onnx file"
        )

    @staticmethod
    def _read_genai_config(model_dir: Path) -> dict[str, Any]:
        config_path = model_dir / "genai_config.json"
        if not config_path.exists():
            eval_logger.warning(
                f"No genai_config.json found in {model_dir}; falling back to defaults."
            )
            return {}
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)

    def _resolve_eot_token_id(self, model_cfg: dict[str, Any]) -> int:
        eos = model_cfg.get("eos_token_id")
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        if eos is not None:
            return int(eos)
        # Fall back to the tokenizer once it is loaded (see _load).
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # Engine hooks (override in subclasses to change engine / EP selection)
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
        """Load the tokenizer and the onnxruntime-genai model."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        if self._eot_token_id is None:
            self._eot_token_id = (
                self.tokenizer.eos_token_id
                if self.tokenizer.eos_token_id is not None
                else 0
            )
        if self._bos_token_id is None:
            self._bos_token_id = self.tokenizer.bos_token_id

        config = self.og.Config(str(self.model_dir))
        self._select_ep(config)
        self.model = self.og.Model(config)
        eval_logger.info(f"Loaded onnxruntime-genai model from {self.model_dir}")

    def _forward_logits(self, tokens: list[int]) -> np.ndarray:
        """Run one forward pass and return per-position logits.

        Args:
            tokens: The full input token sequence.

        Returns:
            A ``[seq_len, vocab]`` array where row ``i`` is the distribution over
            the token at position ``i + 1`` given ``tokens[: i + 1]``.
        """
        params = self.og.GeneratorParams(self.model)
        generator = self.og.Generator(self.model, params)
        generator.append_tokens(np.asarray(tokens, dtype=np.int32))
        logits = np.asarray(generator.get_output("logits"), dtype=np.float32)
        if logits.ndim == 3:  # (batch, seq, vocab)
            logits = logits[0]
        elif logits.ndim != 2:
            raise ValueError(f"Unexpected logits shape from GenAI: {logits.shape}")
        return logits

    def _generate(self, tokens: list[int], gen_kwargs: dict[str, Any]) -> list[int]:
        """Greedily (or with sampling) generate token ids from a prompt.

        Args:
            tokens: The prompt token ids.
            gen_kwargs: Generation options (``max_gen_toks``, ``until``,
                ``temperature``, ``top_p``, ``top_k``, ``do_sample``).

        Returns:
            The newly generated token ids (excluding the prompt).
        """
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

    # ------------------------------------------------------------------ #
    # TemplateLM interface
    # ------------------------------------------------------------------ #
    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def prefix_token_id(self) -> int:
        if self._bos_token_id is not None:
            return int(self._bos_token_id)
        return self._eot_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    def tok_encode(
        self, string: str, add_special_tokens: bool | None = None, **kwargs
    ) -> list[int]:
        if add_special_tokens is None:
            add_special_tokens = False
        return self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []

        for cache_key, context_enc, continuation_enc in tqdm(
            requests, disable=disable_tqdm, desc="Computing log-likelihoods"
        ):
            if not continuation_enc:
                results.append((0.0, True))
                continue

            # Feed the full sequence; keep the last max_length tokens so the
            # continuation is always scored, mirroring HFLM's left-truncation.
            # At least one preceding position is required to score the first
            # continuation token, so cap contlen to leave room for context.
            full = (context_enc + continuation_enc)[-self.max_length :]
            contlen = min(len(continuation_enc), len(full) - 1)
            ctxlen = len(full) - contlen

            logits = self._forward_logits(full)
            # Logits at position i predict token i+1, so the continuation is
            # scored by rows [ctxlen-1 : ctxlen-1+contlen].
            cont_logits = logits[ctxlen - 1 : ctxlen - 1 + contlen]
            log_probs = _log_softmax(cont_logits)

            # Left-truncation may have dropped leading continuation tokens.
            targets = np.asarray(continuation_enc[-contlen:], dtype=np.int64)
            token_ll = log_probs[np.arange(contlen), targets]
            total_ll = float(token_ll.sum())
            is_greedy = bool(np.all(log_probs.argmax(axis=-1) == targets))

            results.append((total_ll, is_greedy))

            if cache_key is not None:
                self.cache_hook.add_partial(
                    "loglikelihood", cache_key, (total_ll, is_greedy)
                )

        return results

    def loglikelihood_rolling(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[float]:
        loglikelihoods: list[float] = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Computing rolling log-likelihoods",
        ):
            rolling_token_windows = [
                (None,) + window
                for window in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            ]

            window_lls = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True
            )
            string_ll = sum(ll for ll, _ in window_lls)
            loglikelihoods.append(string_ll)
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_ll)

        return loglikelihoods

    def generate_until(
        self, requests: list["Instance"], disable_tqdm: bool = False
    ) -> list[str]:
        results: list[str] = []

        for context, gen_kwargs in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm,
            desc="Generating text",
        ):
            gen_kwargs = dict(gen_kwargs)
            until = gen_kwargs.pop("until", None)
            if isinstance(until, str):
                until = [until]
            elif until is None:
                until = []

            prompt_tokens = self.tok_encode(context)
            new_tokens = self._generate(prompt_tokens, gen_kwargs)
            text = self.tok_decode(new_tokens)

            for stop in until:
                if stop:
                    idx = text.find(stop)
                    if idx != -1:
                        text = text[:idx]

            results.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)

        return results


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_sum_exp
