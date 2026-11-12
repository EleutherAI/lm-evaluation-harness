"""Result file discovery should handle both Windows and POSIX paths."""

import pytest

from lm_eval.utils import (
    get_file_task_name,
    get_results_filenames,
    get_sample_results_filenames,
)


@pytest.mark.parametrize("sample_path,expected", [
    ("/tmp/my_results/samples_mmlu_20261112.jsonl", "mmlu"),
    (r"C:\\runs\\batch_one\\samples_mmlu_pro_20261112.jsonl", "mmlu_pro"),
    ("samples_hellaswag_20261112.jsonl", "hellaswag"),
])
def test_sample_task_uses_basename_not_parent_directory(sample_path, expected):
    assert get_file_task_name(sample_path) == expected


def test_result_file_filtering_accepts_windows_paths_and_exact_extensions():
    paths = [
        "/tmp/my_results/results_20261112.json",
        r"C:\\runs\\session_one\\results_20261113.json",
        "results_20261114.json",
        "results_20261115.jsonl",
        "results_20261116.json.bak",
        "/tmp/results_archive/notes.json",
    ]
    assert get_results_filenames(paths) == paths[:3]


def test_sample_file_filtering_accepts_windows_paths_and_exact_extensions():
    paths = [
        "/tmp/my_results/samples_mmlu_20261112.jsonl",
        r"C:\\runs\\session_one\\samples_mmlu_20261113.jsonl",
        "samples_hellaswag_20261114.jsonl",
        "samples_hellaswag_20261115.json",
        "samples_hellaswag_20261116.jsonl.bak",
        "/tmp/samples_archive/notes.jsonl",
    ]
    assert get_sample_results_filenames(paths) == paths[:3]
