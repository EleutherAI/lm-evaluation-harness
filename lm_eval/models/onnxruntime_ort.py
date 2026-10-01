"""Raw ONNX Runtime backend for lm-eval-harness.

Runs a Model Builder ONNX export through a plain ``onnxruntime.InferenceSession``
-- the same runtime a deployment uses -- rather than through the
``onnxruntime-genai`` loop. Two reasons to prefer it:

* **Stack parity.** Scores come from the exact session/EP/kernels a customer
  runs in production, so an eval number is attributable to the deployed stack.
* **Provider reach.** It supports execution providers ``onnxruntime-genai`` does
  not build, notably ROCm and MIGraphX for AMD GPUs.

It shares all lm-eval logic with the ``onnxruntime-genai`` backend via
:class:`lm_eval.models._onnx_base._ONNXLMBase`; only the forward pass differs.

Example usage:
    lm_eval --model onnxruntime --model_args pretrained=path/to/model_builder_output,execution_provider=cuda --tasks hellaswag --limit 10

Scope: this backend implements the log-likelihood path, which covers
``loglikelihood``, ``multiple_choice`` (mmlu/hellaswag/arc), and
``loglikelihood_rolling`` (wikitext perplexity). Generative tasks are not
supported yet -- the KV-cache decode loop is tracked separately.

Graph assumptions are introspected, not hardcoded: ``position_ids`` is absent
from GQA+rotary exports, and past/present KV tensor names follow the printf
templates in ``genai_config.json``. Note that the graph must be loadable by the
installed ``onnxruntime``: Model Builder emits contrib ops whose schemas evolve,
so an ORT older than the builder may reject the graph outright (e.g. a 12-input
``GroupQueryAttention`` fails to load on ORT 1.22).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from lm_eval.api.registry import register_model
from lm_eval.models._onnx_base import _ONNXLMBase


eval_logger = logging.getLogger(__name__)

# Friendly aliases -> onnxruntime execution provider names. Full ORT names are
# accepted as-is, so new providers work without a code change.
_EP_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "qnn": "QNNExecutionProvider",
}

_ORT_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


@register_model("onnxruntime")
class ONNXRuntimeLM(_ONNXLMBase):
    """Model Builder ONNX evaluation through a raw ``InferenceSession``."""

    # ------------------------------------------------------------------ #
    # Engine hooks
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for the onnxruntime backend. Install a "
                "matching EP wheel, e.g. `pip install onnxruntime` (CPU), "
                "`pip install onnxruntime-gpu` (CUDA), or `pip install "
                "onnxruntime-rocm` (ROCm). These wheels are mutually exclusive."
            ) from e

        self.ort = ort
        eval_logger.info(f"ONNX Runtime version: {ort.__version__}")

        graph_path = self._resolve_graph_path()
        providers, provider_options = self._resolve_providers(ort)

        self.session = ort.InferenceSession(
            str(graph_path),
            ort.SessionOptions(),
            providers=providers,
            provider_options=provider_options,
        )
        eval_logger.info(
            f"Loaded {graph_path} with providers {self.session.get_providers()}"
        )
        self._introspect_graph()

    def _forward_logits(self, tokens: list[int]) -> np.ndarray:
        seq_len = len(tokens)
        feeds: dict[str, np.ndarray] = {}

        for name, dtype, shape in self._input_specs:
            if name == "input_ids":
                feeds[name] = np.asarray([tokens], dtype=dtype)
            elif name == "attention_mask":
                feeds[name] = np.ones((1, seq_len), dtype=dtype)
            elif name == "position_ids":
                feeds[name] = np.arange(seq_len, dtype=dtype).reshape(1, seq_len)
            else:
                # Past KV (and any other auxiliary input): start empty, so a
                # single pass scores every position of the prompt.
                concrete = [self._resolve_dim(d, seq_len) for d in shape]
                feeds[name] = np.zeros(concrete, dtype=dtype)

        logits = self.session.run([self._logits_name], feeds)[0]
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 3:  # (batch, seq, vocab)
            logits = logits[0]
        elif logits.ndim != 2:
            raise ValueError(
                f"Unexpected logits shape from ONNX Runtime: {logits.shape}"
            )
        return logits

    def _generate(self, tokens: list[int], gen_kwargs: dict[str, Any]) -> list[int]:
        raise NotImplementedError(
            "The `onnxruntime` backend does not implement generative tasks yet; "
            "it supports loglikelihood, multiple_choice, and loglikelihood_rolling "
            "(e.g. hellaswag, arc, mmlu, wikitext). For generative tasks such as "
            "gsm8k, use `--model onnxruntime-genai` with the same model directory."
        )

    # ------------------------------------------------------------------ #
    # Graph / provider resolution
    # ------------------------------------------------------------------ #
    def _resolve_graph_path(self) -> Path:
        """Locate the ONNX graph: explicit file, manifest filename, then glob."""
        explicit = Path(self.pretrained)
        if explicit.is_file() and explicit.suffix == ".onnx":
            return explicit

        filename = (
            self._genai_config.get("model", {}).get("decoder", {}).get("filename")
        )
        if filename:
            candidate = self.model_dir / filename
            if candidate.is_file():
                return candidate
            eval_logger.warning(
                f"genai_config.json names {filename}, which is missing from "
                f"{self.model_dir}; falling back to a directory scan."
            )

        candidates = sorted(self.model_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"No .onnx graph found in {self.model_dir}")
        if len(candidates) > 1:
            eval_logger.warning(
                f"Multiple ONNX graphs in {self.model_dir} "
                f"({[c.name for c in candidates]}); using {candidates[0].name}. "
                "Pass the .onnx path directly to choose another."
            )
        return candidates[0]

    def _declared_providers(self) -> list[tuple[str, dict[str, Any]]]:
        """Providers the export declares, as ``(name, options)`` pairs.

        ``InferenceSession`` ignores ``genai_config.json``, so we read
        ``model.decoder.session_options.provider_options`` ourselves to keep the
        default behavior aligned with the ``onnxruntime-genai`` backend.
        """
        session_options = (
            self._genai_config.get("model", {})
            .get("decoder", {})
            .get("session_options", {})
        )
        declared: list[tuple[str, dict[str, Any]]] = []
        for entry in session_options.get("provider_options") or []:
            if not isinstance(entry, dict):
                continue
            for name, options in entry.items():
                declared.append((name, dict(options) if options else {}))
        return declared

    def _resolve_providers(self, ort) -> tuple[list[str], list[dict[str, Any]]]:
        """Resolve the ORT provider chain, with CPU appended as fallback.

        Mirrors the ``onnxruntime-genai`` backend: when ``execution_provider`` is
        ``None`` the providers declared in ``genai_config.json`` are honored, so
        a CUDA/DML/NPU export is not silently downgraded to CPU. An explicit
        ``execution_provider`` overrides the export, and ``provider_options``
        applies to that named provider.
        """
        available = ort.get_available_providers()

        def to_ort_name(name: str, source: str) -> str:
            provider = _EP_ALIASES.get(name.lower(), name)
            if provider not in available:
                raise ValueError(
                    f"Execution provider {provider!r} (from {source}) is not "
                    f"available in this onnxruntime build. Available: {available}. "
                    "Install the matching wheel (onnxruntime-gpu for CUDA, "
                    "onnxruntime-rocm for ROCm, ...)."
                )
            return provider

        if self.execution_provider is None:
            declared = self._declared_providers()
            if not declared:
                # genai treats an empty provider list as CPU, so match that.
                eval_logger.info(
                    "No execution_provider given and genai_config.json declares "
                    "none; running on CPU."
                )
                providers = ["CPUExecutionProvider"]
                provider_options: list[dict[str, Any]] = [dict(self.provider_options)]
            else:
                providers = [
                    to_ort_name(name, "genai_config.json") for name, _ in declared
                ]
                provider_options = [options for _, options in declared]
                eval_logger.info(
                    f"Using providers declared in genai_config.json: {providers}"
                )
        else:
            requested = self.execution_provider
            providers = [to_ort_name(requested, f"execution_provider={requested!r}")]
            provider_options = [dict(self.provider_options)]
            eval_logger.info(f"Using execution provider: {providers[0]}")

        if "CPUExecutionProvider" not in providers:
            # Model Builder graphs use contrib ops that some EPs do not
            # implement (e.g. GroupQueryAttention has no ROCm kernel), so keep
            # CPU available for per-node fallback rather than failing to load.
            providers.append("CPUExecutionProvider")
            provider_options.append({})
        return providers, provider_options

    def _introspect_graph(self) -> None:
        """Record input specs and the logits output name from the live session."""
        self._input_specs: list[tuple[str, Any, list[Any]]] = []
        for inp in self.session.get_inputs():
            dtype = _ORT_TO_NUMPY.get(inp.type)
            if dtype is None:
                raise NotImplementedError(
                    f"Input {inp.name!r} has unsupported type {inp.type}. "
                    "bfloat16 KV caches are not supported by this backend yet; "
                    "export with -p fp16 or -p fp32 instead."
                )
            self._input_specs.append((inp.name, dtype, list(inp.shape)))

        input_names = {name for name, _, _ in self._input_specs}
        self._has_position_ids = "position_ids" in input_names

        configured = (
            self._genai_config.get("model", {})
            .get("decoder", {})
            .get("outputs", {})
            .get("logits")
        )
        output_names = [out.name for out in self.session.get_outputs()]
        if configured in output_names:
            self._logits_name = configured
        elif "logits" in output_names:
            self._logits_name = "logits"
        else:
            self._logits_name = output_names[0]
            eval_logger.warning(
                f"No 'logits' output found in {output_names}; "
                f"using {self._logits_name}."
            )

        eval_logger.info(
            f"Graph inputs: {len(self._input_specs)} "
            f"(position_ids={'present' if self._has_position_ids else 'absent'}), "
            f"logits output: {self._logits_name}"
        )

    @staticmethod
    def _resolve_dim(dim: Any, seq_len: int) -> int:
        """Bind a symbolic ONNX dimension for a single prompt pass.

        Past-sequence axes bind to 0 (empty KV cache); batch is 1; remaining
        sequence axes bind to the prompt length. Checked in this order because
        ``past_sequence_length`` also contains ``seq``.
        """
        if isinstance(dim, int):
            return dim
        name = (dim or "").lower()
        if "batch" in name:
            return 1
        if "past" in name:
            return 0
        if "total" in name or "seq" in name:
            return seq_len
        return 1
