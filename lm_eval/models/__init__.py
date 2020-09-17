from .gpt2 import GPT2LM
from .gpt3 import GPT3LM

MODEL_REGISTRY = {
    "gpt2": GPT2LM,
    "gpt3": GPT3LM,
}


def get_model(model_name):
    return MODEL_REGISTRY[model_name]
