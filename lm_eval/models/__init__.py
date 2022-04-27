from . import gpt2
from . import gptj
from . import gpt3
from . import t5
from . import t0
from . import dummy

MODEL_REGISTRY = {
    "hf": gpt2.HFLM,
    "gpt2": gpt2.GPT2LM,
    "gptj": gptj.GPTJLM,
    "gpt3": gpt3.GPT3LM,
    "t5": t5.T5LM,
    "mt5": t5.T5LM,
    "t0": t0.T0LM,
    "dummy": dummy.DummyLM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
