import importlib.metadata
import logging
import os
from importlib.util import find_spec


__version__ = importlib.metadata.version("lm_eval")


# Enable high-performance transfers
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")  # huggingface_hub >= 0.32.0
if find_spec("hf_transfer") is not None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # legacy hf_transfer


# Lazy-load .evaluator module to improve CLI startup
def __getattr__(name):
    if name == "evaluate":
        from .evaluator import evaluate

        return evaluate
    elif name == "simple_evaluate":
        from .evaluator import simple_evaluate

        return simple_evaluate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["evaluate", "simple_evaluate", "__version__"]
