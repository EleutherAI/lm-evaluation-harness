"""
LiveCodeBenchmark evaluation utilities.

Evaluates code generation on the LiveCodeBench dataset using:
- pass@1: Does the first generated code pass all test cases?
- pass@5: Do any of the top 5 generated codes pass all test cases?

Inspired by the APPS benchmark evaluation approach.
"""

import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List


def extract_test_cases(doc: Dict[str, Any]) -> tuple:
    """
    Extract test cases from the LiveCodeBench dataset entry.

    Returns:
        (test_list, entry_point, timeout)
        - test_list: list of test input/output pairs
        - entry_point: function name to call
        - timeout: execution timeout in seconds
    """
    # LiveCodeBench format: tests is a list of dicts with 'input' and 'output'
    test_list = doc.get("tests", [])
    entry_point = doc.get("entry_point", "solution")
    timeout = doc.get("timeout", 10)

    return test_list, entry_point, timeout


def run_code(code: str, test_input: str, entry_point: str, timeout: int = 10) -> tuple:
    """
    Execute generated code against a test case.

    Args:
        code: Generated Python code
        test_input: Input to the function
        entry_point: Function name to call
        timeout: Execution timeout in seconds

    Returns:
        (success, output)
        - success: True if code executed without error
        - output: stdout/stderr output or error message
    """
    try:
        # Create a temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Wrap the user code with test execution
            test_script = f"""
{code}

# Execute the entry point with the test input
import json
try:
    input_data = json.loads({repr(test_input)})
    if input_data is None:
        result = {entry_point}()
    elif isinstance(input_data, list):
        result = {entry_point}(*input_data)
    else:
        result = {entry_point}(input_data)
    print(json.dumps(result))
except Exception as e:
    print(f"ERROR: {{e}}", flush=True)
"""
            f.write(test_script)
            temp_file = f.name

        # Execute the code with timeout
        try:
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout.strip()

            # Check if there was an error
            if result.returncode != 0 or output.startswith("ERROR:"):
                return False, result.stderr.strip() or output

            return True, output

        except subprocess.TimeoutExpired:
            return False, f"Timeout (>{timeout}s)"

        finally:
            os.unlink(temp_file)

    except Exception as e:
        return False, f"Execution error: {str(e)}"


def check_correctness(generated_output: str, expected_output: str) -> bool:
    """
    Check if generated output matches expected output.

    Handles various formats (JSON, numbers, strings) with lenient comparison.
    """
    try:
        # Try JSON comparison first (for structured output)
        import json
        gen = json.loads(generated_output)
        exp = json.loads(expected_output)
        if gen == exp:
            return True
    except Exception:
        pass

    # String comparison with normalization
    gen_str = str(generated_output).strip().lower()
    exp_str = str(expected_output).strip().lower()

    # Exact match
    if gen_str == exp_str:
        return True

    # Numeric comparison (for floating point)
    try:
        gen_num = float(generated_output)
        exp_num = float(expected_output)
        return abs(gen_num - exp_num) < 1e-4
    except Exception:
        pass

    return False


def evaluate_code(code: str, test_cases: List[Dict], entry_point: str, timeout: int = 10) -> bool:
    """
    Check if generated code passes all test cases.

    Args:
        code: Generated Python code
        test_cases: List of {"input": ..., "output": ...} dicts
        entry_point: Function name to call
        timeout: Execution timeout per test case

    Returns:
        True if code passes ALL test cases, False otherwise
    """
    # Must pass all test cases
    for test_case in test_cases:
        test_input = test_case.get("input", "")
        expected_output = test_case.get("output", "")

        success, output = run_code(code, test_input, entry_point, timeout)

        if not success or not check_correctness(output, expected_output):
            return False

    return True


def compute_pass_at_k(results: List[bool], k: int) -> float:
    """
    Compute pass@k metric.

    pass@k = 1 - (number of samples that fail on all k attempts) / (total samples)

    For a single sample: pass@k = 1 if any of the k outputs is correct, 0 otherwise
    """
    if not results:
        return 0.0

    # For a single problem: pass@k = 1 if any of the top k codes passed
    return 1.0 if any(results[:k]) else 0.0


def process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    """
    Main evaluation function called by lm-evaluation-harness.

    Args:
        doc: Single dataset example with "tests", "entry_point", etc.
        results: List of generated code snippets (typically 1 for pass@1, up to 5 for pass@5)

    Returns:
        Dict with "pass@1" and "pass@5" metrics
    """
    assert len(results) >= 1, "Must have at least 1 generated code"

    # Extract test cases from the problem
    test_cases, entry_point, timeout = extract_test_cases(doc)

    # Evaluate each generated code
    pass_list = []
    for code in results:
        passed = evaluate_code(code, test_cases, entry_point, timeout)
        pass_list.append(passed)

    # Compute pass@1 and pass@5
    pass_at_1 = compute_pass_at_k(pass_list, k=1)
    pass_at_5 = compute_pass_at_k(pass_list, k=min(5, len(pass_list)))

    return {
        "pass@1": pass_at_1,
        "pass@5": pass_at_5,
    }
