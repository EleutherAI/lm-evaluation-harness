# ruff: noqa
def pytest_ignore_collect(path, config):
    """Return True to prevent pytest from collecting problematic model tests"""
    try:
        from transformers import EncoderDecoderCache

        return False  # Allow collection if import succeeds
    except ImportError:
        # Only ignore files in the models directory
        if "models/test_" in str(path):
            return True  # Skip collection
    return False  # Collect all other tests
