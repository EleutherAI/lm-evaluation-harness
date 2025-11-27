import json
import re

import pytest


pytest.importorskip("zeno_client")
from scripts.zeno_visualize import sanitize_string


def test_zeno_sanitize_string():
    """
    Test that the model_args handling logic in zeno_visualize.py properly handles
    different model_args formats (string and dictionary).
    """

    # Define the process_model_args function that replicates the fixed logic in zeno_visualize.py
    # Test case 1: model_args as a string
    string_model_args = "pretrained=EleutherAI/pythia-160m,dtype=float32"
    result_string = sanitize_string(string_model_args)
    expected_string = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", string_model_args)

    # Test case 2: model_args as a dictionary
    dict_model_args = {"pretrained": "EleutherAI/pythia-160m", "dtype": "float32"}
    result_dict = sanitize_string(dict_model_args)
    expected_dict = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", json.dumps(dict_model_args))

    # Verify the results
    assert result_string == expected_string
    assert result_dict == expected_dict

    # Also test that the sanitization works as expected
    assert ":" not in result_string  # No colons in sanitized output
    assert ":" not in result_dict  # No colons in sanitized output
    assert "/" not in result_dict  # No slashes in sanitized output
    assert "<" not in result_dict  # No angle brackets in sanitized output


if __name__ == "__main__":
    test_zeno_sanitize_string()
    print("All tests passed.")
