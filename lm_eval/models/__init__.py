from . import gpt2
from . import gpt3
from . import textsynth
from . import dummy

MODEL_REGISTRY = {
    "hf-causal": gpt2.HFLM,
    "openai": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
