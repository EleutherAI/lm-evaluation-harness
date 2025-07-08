from . import (
    anthropic_llms,
    api_models,
    dummy,
    gguf,
    hf_audiolm,
    hf_steered,
    hf_vlms,
    huggingface,
    ibm_watsonx_ai,
    mamba_lm,
    nemo_lm,
    neuron_optimum,
    openai_completions,
    optimum_ipex,
    optimum_lm,
    sglang_causallms,
    sglang_generate_API,
    textsynth,
    vllm_causallms,
    vllm_vlms,
)


# TODO: implement __all__


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
