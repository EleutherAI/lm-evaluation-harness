# ruff: noqa
import pytest


try:
    from transformers import EncoderDecoderCache
except ImportError:
    pytest.skip(
        "transformers.EncoderDecoderCache is required for model tests",
        allow_module_level=True,
    )
