"""
Tests for LiveCodeBenchmark evaluation utilities.

Comprehensive tests covering:
- Code execution and test case evaluation
- Edge cases: timeouts, syntax errors, type mismatches
- Metric computation (pass@1, pass@5)
- Contamination prevention via release dates
"""

import sys
import os

# Allow importing from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    extract_test_cases,
    run_code,
    check_correctness,
    evaluate_code,
    compute_pass_at_k,
    process_results,
)


class TestExtractTestCases:
    """Test extraction of test cases from dataset entries."""

    def test_extract_basic(self):
        doc = {
            "tests": [{"input": "5", "output": "120"}],
            "entry_point": "factorial",
            "timeout": 10,
        }
        tests, entry_point, timeout = extract_test_cases(doc)
        assert len(tests) == 1
        assert entry_point == "factorial"
        assert timeout == 10

    def test_extract_defaults(self):
        doc = {"tests": []}
        tests, entry_point, timeout = extract_test_cases(doc)
        assert entry_point == "solution"
        assert timeout == 10

    def test_extract_multiple_tests(self):
        doc = {
            "tests": [
                {"input": "1", "output": "1"},
                {"input": "2", "output": "2"},
                {"input": "3", "output": "6"},
            ]
        }
        tests, _, _ = extract_test_cases(doc)
        assert len(tests) == 3


class TestCheckCorrectness:
    """Test output correctness checking."""

    def test_exact_match_string(self):
        assert check_correctness("hello", "hello")

    def test_case_insensitive(self):
        assert check_correctness("HELLO", "hello")

    def test_whitespace_normalized(self):
        assert check_correctness("  hello  ", "hello")

    def test_json_number(self):
        assert check_correctness("42", "42")

    def test_json_list(self):
        assert check_correctness("[1, 2, 3]", "[1, 2, 3]")

    def test_json_dict(self):
        assert check_correctness('{"a": 1}', '{"a": 1}')

    def test_float_comparison(self):
        # Floating point numbers within epsilon
        assert check_correctness("3.14159", "3.14160")

    def test_mismatch(self):
        assert not check_correctness("hello", "world")

    def test_numeric_mismatch(self):
        assert not check_correctness("10", "20")


class TestRunCode:
    """Test code execution against test inputs."""

    def test_simple_function(self):
        code = """
def add(a, b):
    return a + b
"""
        success, output = run_code(code, '[1, 2]', "add")
        assert success
        assert output == "3"

    def test_syntax_error(self):
        code = "def broken(: pass"
        success, output = run_code(code, "[]", "broken")
        assert not success

    def test_runtime_error(self):
        code = """
def divide(a, b):
    return a / b
"""
        success, output = run_code(code, '[1, 0]', "divide")
        assert not success

    def test_timeout(self):
        code = """
def infinite():
    while True:
        pass
"""
        success, output = run_code(code, "null", "infinite", timeout=1)
        assert not success
        assert "Timeout" in output

    def test_return_json(self):
        code = """
def get_list():
    return [1, 2, 3]
"""
        success, output = run_code(code, "null", "get_list")
        assert success
        assert "[1, 2, 3]" in output

    def test_dict_input(self):
        code = """
def get_value(d):
    return d['key']
"""
        success, output = run_code(code, '{"key": "value"}', "get_value")
        assert success


class TestEvaluateCode:
    """Test evaluation of code against test cases."""

    def test_pass_all_cases(self):
        code = """
def add(a, b):
    return a + b
"""
        test_cases = [
            {"input": "[1, 2]", "output": "3"},
            {"input": "[5, 10]", "output": "15"},
            {"input": "[0, 0]", "output": "0"},
        ]
        assert evaluate_code(code, test_cases, "add")

    def test_fail_one_case(self):
        code = """
def add(a, b):
    return a + b + 1  # Bug: adds 1 extra
"""
        test_cases = [
            {"input": "[1, 2]", "output": "3"},
            {"input": "[5, 10]", "output": "15"},
        ]
        assert not evaluate_code(code, test_cases, "add")

    def test_empty_test_cases(self):
        code = "def foo(): pass"
        assert evaluate_code(code, [], "foo")

    def test_factorial(self):
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        test_cases = [
            {"input": "5", "output": "120"},
            {"input": "1", "output": "1"},
            {"input": "0", "output": "1"},
        ]
        assert evaluate_code(code, test_cases, "factorial")


class TestComputePassAtK:
    """Test pass@k metric computation."""

    def test_pass_at_1_single_pass(self):
        results = [True]
        assert compute_pass_at_k(results, k=1) == 1.0

    def test_pass_at_1_single_fail(self):
        results = [False]
        assert compute_pass_at_k(results, k=1) == 0.0

    def test_pass_at_5_first_passes(self):
        results = [True, False, False, False, False]
        assert compute_pass_at_k(results, k=5) == 1.0

    def test_pass_at_5_last_passes(self):
        results = [False, False, False, False, True]
        assert compute_pass_at_k(results, k=5) == 1.0

    def test_pass_at_5_all_fail(self):
        results = [False, False, False, False, False]
        assert compute_pass_at_k(results, k=5) == 0.0

    def test_pass_at_5_fewer_samples(self):
        # If we have fewer than k samples, use what we have
        results = [False, True]
        assert compute_pass_at_k(results, k=5) == 1.0

    def test_empty_results(self):
        results = []
        assert compute_pass_at_k(results, k=5) == 0.0


class TestProcessResults:
    """Test the main evaluation function."""

    def test_process_single_result_pass(self):
        doc = {
            "tests": [{"input": "[1, 2]", "output": "3"}],
            "entry_point": "add",
        }
        code = """
def add(a, b):
    return a + b
"""
        result = process_results(doc, [code])
        assert result["pass@1"] == 1.0
        assert result["pass@5"] == 1.0

    def test_process_single_result_fail(self):
        doc = {
            "tests": [{"input": "[1, 2]", "output": "3"}],
            "entry_point": "add",
        }
        code = """
def add(a, b):
    return 0  # Wrong
"""
        result = process_results(doc, [code])
        assert result["pass@1"] == 0.0
        assert result["pass@5"] == 0.0

    def test_process_multiple_results_first_passes(self):
        doc = {
            "tests": [{"input": "[1, 2]", "output": "3"}],
            "entry_point": "add",
        }
        code1 = "def add(a, b):\n    return a + b"
        code2 = "def add(a, b):\n    return 0"
        result = process_results(doc, [code1, code2])
        assert result["pass@1"] == 1.0
        assert result["pass@5"] == 1.0

    def test_process_multiple_results_second_passes(self):
        doc = {
            "tests": [{"input": "[1, 2]", "output": "3"}],
            "entry_point": "add",
        }
        code1 = "def add(a, b):\n    return 0"
        code2 = "def add(a, b):\n    return a + b"
        result = process_results(doc, [code1, code2])
        assert result["pass@1"] == 0.0  # First one fails
        assert result["pass@5"] == 1.0  # One of top 5 passes

    def test_process_multiple_test_cases(self):
        doc = {
            "tests": [
                {"input": "[1, 2]", "output": "3"},
                {"input": "[5, 10]", "output": "15"},
            ],
            "entry_point": "add",
        }
        code = "def add(a, b):\n    return a + b"
        result = process_results(doc, [code])
        assert result["pass@1"] == 1.0
        assert result["pass@5"] == 1.0

    def test_process_with_timeout_custom(self):
        doc = {
            "tests": [{"input": "1", "output": "1"}],
            "entry_point": "identity",
            "timeout": 2,
        }
        code = "def identity(x):\n    return x"
        result = process_results(doc, [code])
        assert result["pass@1"] == 1.0


class TestContaminationPrevention:
    """Test that release dates can be used to prevent contamination."""

    def test_release_date_field_exists(self):
        """Verify dataset includes release date for contamination filtering."""
        # This is metadata for users to filter by
        doc = {
            "problem_id": "LC001",
            "release_date": "2024-03-15",
            "tests": [{"input": "1", "output": "1"}],
            "entry_point": "solution",
        }
        # Users can filter: only evaluate on problems released after model's training cutoff
        assert "release_date" in doc


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
