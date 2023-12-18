from . import gpt2
from . import gpt3
from . import anthropic_llms
from . import huggingface
from . import textsynth
from . import deepsparse
from . import dummy
from . import gguf

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "hf-causal": gpt2.HFLM,
    "hf-causal-experimental": huggingface.AutoCausalLM,
    "hf-seq2seq": huggingface.AutoSeq2SeqLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "anthropic": anthropic_llms.AnthropicLM,
    "textsynth": textsynth.TextSynthLM,
    "deepsparse": deepsparse.DeepSparseLM,
    "dummy": dummy.DummyLM,
    "gguf": gguf.GGUFLM,
    "optimum-causal": gpt2.OPTIMUMLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
