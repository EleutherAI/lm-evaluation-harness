import importlib.metadata
import logging
import os


__version__ = importlib.metadata.version("lm_eval")


# Enable hf_transfer if available
try:
    import hf_transfer  # type: ignore
    import huggingface_hub.constants  # type: ignore

    huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
except ImportError:
    pass


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
