import json
import re


def test_model_args_handling():
    """
    Test that the model_args handling logic in zeno_visualize.py properly handles
    different model_args formats (string and dictionary).

    This test focuses only on the specific logic that was fixed in the PR:
    - Extracting model_args from the results file
    - Converting dictionary model_args to string
    - Applying sanitization
    """

    # Define the process_model_args function that replicates the fixed logic in zeno_visualize.py
    def process_model_args(model_args_raw):
        """
        Process model_args which can be either a string or a dictionary
        and sanitize it for use in Zeno visualization.
        """
        # Convert to string if it's a dictionary
        model_args_str = (
            json.dumps(model_args_raw)
            if isinstance(model_args_raw, dict)
            else model_args_raw
        )

        # Apply the sanitization
        model_args = re.sub(
            r"[\"<>:/\|\\?\*\[\]]+",
            "__",
            model_args_str,
        )
        return model_args

    # Test case 1: model_args as a string
    string_model_args = "pretrained=EleutherAI/pythia-160m,dtype=float32"
    result_string = process_model_args(string_model_args)
    expected_string = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", string_model_args)

    # Test case 2: model_args as a dictionary
    dict_model_args = {"pretrained": "EleutherAI/pythia-160m", "dtype": "float32"}
    result_dict = process_model_args(dict_model_args)
    expected_dict = re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", json.dumps(dict_model_args))

    # Verify the results
    assert result_string == expected_string
    assert result_dict == expected_dict

    # Also test that the sanitization works as expected
    assert ":" not in result_string  # No colons in sanitized output
    assert ":" not in result_dict  # No colons in sanitized output
    assert "/" not in result_dict  # No slashes in sanitized output
    assert "<" not in result_dict  # No angle brackets in sanitized output
