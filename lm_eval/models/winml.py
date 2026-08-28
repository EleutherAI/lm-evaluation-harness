"""WinML backend for lm-eval-harness with NPU/GPU/CPU support.

This backend runs Model Builder ONNX models through the Windows Machine Learning
(WinML) execution-provider catalog. It is a thin specialization of
:class:`~lm_eval.models.onnxruntime_genai.ONNXRuntimeGenAILM`: all scoring and
generation logic is inherited, and only execution-provider selection is
Windows-specific.

Example usage:
    lm_eval --model winml --model_args pretrained=path/to/onnx/model --tasks hellaswag

"""

import logging
from pathlib import Path

from lm_eval.api.registry import register_model
from lm_eval.models.onnxruntime_genai import ONNXRuntimeGenAILM


eval_logger = logging.getLogger(__name__)


@register_model("winml")
class WindowsML(ONNXRuntimeGenAILM):
    """WindowsML backend: onnxruntime-genai with Windows ML provider selection.

    Inherits all lm-eval logic (tokenization, single-pass log-likelihood,
    rolling perplexity, and generation) from
    :class:`~lm_eval.models.onnxruntime_genai.ONNXRuntimeGenAILM`, overriding
    only :meth:`_select_ep` to register execution providers through the Windows
    ML catalog.

    The default ``max_length`` / ``max_gen_toks`` are pinned to the historical
    winml values so numeric behavior is unchanged; the cross-platform backend
    picks its own (more model-aware) defaults.
    """

    # Historical winml defaults, kept so existing winml runs are unaffected.
    _WINML_MAX_LENGTH = 4096
    _WINML_MAX_GEN_TOKS = 4096

    def __init__(
        self,
        pretrained: str,
        max_length: int | None = None,
        max_gen_toks: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained,
            max_length=max_length or self._WINML_MAX_LENGTH,
            max_gen_toks=max_gen_toks or self._WINML_MAX_GEN_TOKS,
            **kwargs,
        )

    def _select_ep(self, config) -> None:
        """Register Windows ML execution providers on the ``og.Config``.

        Discovers providers via the Windows ML catalog and registers them with
        onnxruntime-genai. If discovery fails (e.g. running outside Windows), we
        fall back to the cross-platform selection in the base class.
        """
        self._fix_winrt_runtime()
        if self._register_winml_providers_to_genai():
            self._log_winml_devices()
        else:
            eval_logger.warning(
                "Windows ML provider registration failed; falling back to "
                "cross-platform execution-provider selection."
            )
        super()._select_ep(config)

    def _fix_winrt_runtime(self) -> None:
        """Remove the bundled msvcp140.dll from winrt-runtime to avoid clashes.

        This DLL, shipped by the ``winrt-runtime`` package, can conflict with
        other libraries loaded in the same process.
        """
        try:
            from importlib import metadata
        except ImportError:
            return
        try:
            site_packages_path = Path(
                str(metadata.distribution("winrt-runtime").locate_file(""))
            )
        except metadata.PackageNotFoundError:
            return
        dll_path = site_packages_path / "winrt" / "msvcp140.dll"
        if dll_path.exists():
            dll_path.unlink()

    def _register_winml_providers_to_genai(self) -> bool:
        """Discover Windows ML providers and register them with onnxruntime-genai.

        Returns:
            True if registration succeeded, False otherwise.
        """
        try:
            import winui3.microsoft.windows.ai.machinelearning as winml
            from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
                InitializeOptions,
                initialize,
            )

            with initialize(options=InitializeOptions.ON_NO_MATCH_SHOW_UI):
                catalog = winml.ExecutionProviderCatalog.get_default()
                providers = catalog.find_all_providers()
                for provider in providers:
                    provider.ensure_ready_async().get()
                    self.og.register_execution_provider_library(
                        provider.name, provider.library_path
                    )
                    eval_logger.info(
                        "Registered %s to ONNX Runtime GenAI", provider.name
                    )
            return True
        except ImportError as e:
            eval_logger.warning("Windows ML import error: %s", e)
            return False
        except Exception as e:  # noqa: BLE001
            eval_logger.warning("Error registering providers to GenAI: %s", e)
            return False

    def _log_winml_devices(self) -> None:
        """Log available execution-provider devices via the Windows ML API."""
        try:
            import onnxruntime as ort

            ep_devices = ort.get_ep_devices()
            ep_device_map: dict[str, list] = {}
            for device in ep_devices:
                ep_device_map.setdefault(device.ep_name, []).append(device)

            eval_logger.info(
                "Available EP devices: %s execution providers", len(ep_device_map)
            )
            for name, devices in ep_device_map.items():
                eval_logger.info("Execution Provider: %s", name)
                for device in devices:
                    try:
                        device_type = ort.OrtHardwareDeviceType(device.device.type).name
                    except Exception:  # noqa: BLE001
                        device_type = "Unknown"
                    eval_logger.info(
                        " | Vendor: %-16s | Device Type: %-8s",
                        device.ep_vendor,
                        device_type,
                    )
        except Exception as e:  # noqa: BLE001
            eval_logger.warning("Windows ML device enumeration failed: %s", e)
