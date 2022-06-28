from . import dummy
from . import openai_completions
from . import huggingface


MODEL_REGISTRY = {
    "hf-causal": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "openai": openai_completions.OpenAICompletionsLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
