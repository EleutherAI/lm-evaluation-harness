from . import (
    anthropic_llms,
    dummy,
    gguf,
    huggingface,
    mamba_lm,
    nemo_lm,
    neuron_optimum,
    openai_completions,
    optimum_lm,
    textsynth,
    vllm_causallms,
)


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
