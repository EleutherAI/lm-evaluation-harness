import json
import re

import pytest


pytest.importorskip("zeno_client")
from scripts.zeno_visualize import (
    discover_model_dirs,
    get_model_name,
    sanitize_string,
    tasks_for_model,
)


def _write_results(model_dir, timestamp, tasks):
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {"model_args": "pretrained=EleutherAI/pythia-160m,dtype=float32"},
        "configs": {
            task: {"metric_list": [], "output_type": "multiple_choice"}
            for task in tasks
        },
    }
    results_path = model_dir / f"results_{timestamp}.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    return results_path


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


def test_zeno_discovers_nested_model_dirs(tmp_path):
    data_path = tmp_path / "output"
    pretraining_dir = data_path / "pretraining" / "itskoma__GPT2.5"
    posttraining_dir = data_path / "posttraining" / "itskoma__GPT2.5"
    _write_results(pretraining_dir, "2026-03-20T23-23-24.601482", ["arc_easy", "piqa"])
    _write_results(posttraining_dir, "2026-03-20T23-40-23.244159", ["arc_easy", "piqa"])

    model_dirs = discover_model_dirs(data_path)

    assert {path.relative_to(data_path).as_posix() for path in model_dirs} == {
        "posttraining/itskoma__GPT2.5",
        "pretraining/itskoma__GPT2.5",
    }
    assert {get_model_name(path, data_path) for path in model_dirs} == {
        "posttraining__itskoma__GPT2.5",
        "pretraining__itskoma__GPT2.5",
    }
    assert all("/" not in get_model_name(path, data_path) for path in model_dirs)
    assert tasks_for_model(pretraining_dir) == ["arc_easy", "piqa"]


if __name__ == "__main__":
    test_zeno_sanitize_string()
    print("All tests passed.")
