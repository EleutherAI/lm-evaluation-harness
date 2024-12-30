# ruff: noqa
import pytest


# Check if the required component exists
has_encoder_decoder_cache = False
try:
    from transformers import EncoderDecoderCache

    has_encoder_decoder_cache = True
except ImportError:
    pass


# Mark all tests in this directory as requiring encoder_decoder_cache
def pytest_collection_modifyitems(items):
    skip_marker = pytest.mark.skip(reason="requires transformers.EncoderDecoderCache")
    for item in items:
        if not has_encoder_decoder_cache:
            item.add_marker(skip_marker)
