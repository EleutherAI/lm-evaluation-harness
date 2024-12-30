# # ruff: noqa
# from pathlib import Path
#
# def pytest_ignore_collect(collection_path: Path, config):
#     """Return True to prevent pytest from collecting problematic model tests"""
#     try:
#         from transformers import EncoderDecoderCache
#         return False  # Allow collection if import succeeds
#     except ImportError as e:
#         # Only ignore files in the models directory
#         if "models/test_" in str(collection_path):
#             return True  # Skip collection
#     return False  # Collect all other tests
