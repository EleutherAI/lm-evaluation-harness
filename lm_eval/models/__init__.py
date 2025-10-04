# Models are now lazily loaded via the registry system
# No need to import them all at once

# Define model mappings for lazy registration
MODEL_MAPPING = {
    "anthropic-completions": "lm_eval.models.anthropic_llms:AnthropicLM",
    "anthropic-chat": "lm_eval.models.anthropic_llms:AnthropicChatLM",
    "anthropic-chat-completions": "lm_eval.models.anthropic_llms:AnthropicCompletionsLM",
    "local-completions": "lm_eval.models.openai_completions:LocalCompletionsAPI",
    "local-chat-completions": "lm_eval.models.openai_completions:LocalChatCompletion",
    "openai-completions": "lm_eval.models.openai_completions:OpenAICompletionsAPI",
    "openai-chat-completions": "lm_eval.models.openai_completions:OpenAIChatCompletion",
    "dummy": "lm_eval.models.dummy:DummyLM",
    "gguf": "lm_eval.models.gguf:GGUFLM",
    "ggml": "lm_eval.models.gguf:GGUFLM",
    "hf-audiolm-qwen": "lm_eval.models.hf_audiolm:HFAudioLM",
    "steered": "lm_eval.models.hf_steered:SteeredHF",
    "hf-multimodal": "lm_eval.models.hf_vlms:HFMultimodalLM",
    "hf-auto": "lm_eval.models.huggingface:HFLM",
    "hf": "lm_eval.models.huggingface:HFLM",
    "huggingface": "lm_eval.models.huggingface:HFLM",
    "watsonx_llm": "lm_eval.models.ibm_watsonx_ai:IBMWatsonxAI",
    "mamba_ssm": "lm_eval.models.mamba_lm:MambaLMWrapper",
    "nemo_lm": "lm_eval.models.nemo_lm:NeMoLM",
    "neuronx": "lm_eval.models.neuron_optimum:NeuronModelForCausalLM",
    "ipex": "lm_eval.models.optimum_ipex:IPEXForCausalLM",
    "openvino": "lm_eval.models.optimum_lm:OptimumLM",
    "sglang": "lm_eval.models.sglang_causallms:SGLANG",
    "sglang-generate": "lm_eval.models.sglang_generate_API:SGAPI",
    "textsynth": "lm_eval.models.textsynth:TextSynthLM",
    "vllm": "lm_eval.models.vllm_causallms:VLLM",
    "vllm-vlm": "lm_eval.models.vllm_vlms:VLLM_VLM",
}


# Register all models lazily
def _register_all_models():
    """Register all known models lazily in the registry."""
    from lm_eval.api.registry import model_registry

    for name, path in MODEL_MAPPING.items():
        # Only register if not already present (avoids conflicts when modules are imported)
        if name not in model_registry:
            # Register the lazy placeholder using lazy parameter
            model_registry.register(name, lazy=path)


# Call registration on module import
_register_all_models()

__all__ = ["MODEL_MAPPING"]


try:
    # enable hf hub transfer if available
    import hf_transfer  # type: ignore # noqa
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass
