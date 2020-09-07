import importlib
import os
from ..base import Registry

MODEL_REGISTRY = Registry(registry_name="models")
# Load all modules in models directory to populate registry
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        module_name = file[:file.find('.py')] if file.endswith('.py') else file
        module = importlib.import_module('lm_evaluation_harness.models.' + module_name)


def get_model(model_name):
    return MODEL_REGISTRY.registry[model_name]
