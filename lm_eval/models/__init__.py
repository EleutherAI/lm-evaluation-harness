"""Model implementations for lm_eval.

Models are lazily loaded via the registry system to improve startup performance.

Usage
-----
For programmatic access, use the registry:

    from lm_eval.api.registry import get_model
    model_cls = get_model("hf")
    model = model_cls(pretrained="gpt2")

For direct imports (e.g., subclassing), use explicit module paths:

    from lm_eval.models.huggingface import HFLM
    from lm_eval.models.vllm_causallms import VLLM

Adding New Models
-----------------
1. Create your model class in a new file under lm_eval/models/
2. Use the @register_model decorator on your class
3. Add an entry to MODEL_MAPPING below for lazy discovery
"""

MODEL_MAPPING = {
    "anthropic-chat": "lm_eval.models.anthropic_llms:AnthropicChat",
    "anthropic-chat-completions": "lm_eval.models.anthropic_llms:AnthropicChat",
    "anthropic-completions": "lm_eval.models.anthropic_llms:AnthropicLM",
    "dummy": "lm_eval.models.dummy:DummyLM",
    "ggml": "lm_eval.models.gguf:GGUFLM",
    "gguf": "lm_eval.models.gguf:GGUFLM",
    "hf": "lm_eval.models.huggingface:HFLM",
    "hf-audiolm-qwen": "lm_eval.models.hf_audiolm:HFAudioLM",
    "hf-auto": "lm_eval.models.huggingface:HFLM",
    "hf-mistral3": "lm_eval.models.mistral3:Mistral3LM",
    "hf-multimodal": "lm_eval.models.hf_vlms:HFMultimodalLM",
    "huggingface": "lm_eval.models.huggingface:HFLM",
    "ipex": "lm_eval.models.optimum_ipex:IPEXForCausalLM",
    "local-chat-completions": "lm_eval.models.openai_completions:LocalChatCompletion",
    "local-completions": "lm_eval.models.openai_completions:LocalCompletionsAPI",
    "mamba_ssm": "lm_eval.models.mamba_lm:MambaLMWrapper",
    "megatron_lm": "lm_eval.models.megatron_lm:MegatronLMEval",
    "nemo_lm": "lm_eval.models.nemo_lm:NeMoLM",
    "neuronx": "lm_eval.models.neuron_optimum:NeuronModelForCausalLM",
    "openai-chat-completions": "lm_eval.models.openai_completions:OpenAIChatCompletion",
    "openai-completions": "lm_eval.models.openai_completions:OpenAICompletionsAPI",
    "openvino": "lm_eval.models.optimum_lm:OptimumLM",
    "sglang": "lm_eval.models.sglang_causallms:SGLangLM",
    "sglang-generate": "lm_eval.models.sglang_generate_API:SGLANGGENERATEAPI",
    "steered": "lm_eval.models.hf_steered:SteeredHF",
    "textsynth": "lm_eval.models.textsynth:TextSynthLM",
    "vllm": "lm_eval.models.vllm_causallms:VLLM",
    "vllm-vlm": "lm_eval.models.vllm_vlms:VLLM_VLM",
    "watsonx_llm": "lm_eval.models.ibm_watsonx_ai:IBMWatsonxAI",
    "winml": "lm_eval.models.winml:WindowsML",
}


def _register_all_models():
    """Register all known models lazily in the registry."""
    from lm_eval.api.registry import model_registry

    for name, path in MODEL_MAPPING.items():
        # Only register if not already present (avoids conflicts when modules are imported)
        if name not in model_registry:
            # Register the lazy placeholder
            model_registry.register(name, target=path)


# Call registration on module import
_register_all_models()

__all__ = ["MODEL_MAPPING"]
