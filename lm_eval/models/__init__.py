from . import (
    anthropic_llms,
    api_models,
    dummy,
    gguf,
    ibm_watsonx_ai,
    mamba_lm,
    nemo_lm,
    neuralmagic,
    neuron_optimum,
    openai_completions,
    textsynth,
    vllm_causallms,
    vllm_vlms,
)


__all__ = [
    "anthropic_llms",
    "dummy",
    "gguf",
    "ibm_watsonx_ai",
    "openai_completions",
    "textsynth",
    "vllm_causallms",
]


# try importing all modules that need torch
import importlib


for module_that_needs_torch in [
    "hf_vlms",
    "huggingface",
    "mamba_lm",
    "nemo_lm",
    "neuralmagic",
    "neuron_optimum",
    "optimum_lm",
]:
    try:
        importlib.import_module(f".{module_that_needs_torch}", __name__)
        __all__.append(module_that_needs_torch)
    except ImportError:
        pass


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
