from . import gpt2
from . import gpt3
from . import textsynth
from . import dummy
from . import morphgpt
from . import linggpt

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gpt3": gpt3.GPT3LM,
    "textsynth": textsynth.TextSynthLM,
    "dummy": dummy.DummyLM,
    "morph": morphgpt.MorphGPT,
    "ling": linggpt.LingGPT
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
