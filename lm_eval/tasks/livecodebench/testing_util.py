# lm_eval/tasks/livecodebench/testing_util.py
import json
import sys
import io
import signal
from typing import List, Tuple, Dict, Any

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

def run_test(sample, test, debug=False, timeout=6):
    """
    Execute code and test against expected outputs.
    
    Args:
        sample: Problem sample containing input_output data
        test: Generated code to execute
        debug: Whether to print debug information
        timeout: Timeout in seconds for each test case
    
    Returns:
        Tuple of (results_list, metadata_dict)
    """
    try:
        # Parse input_output data
        input_output_str = sample.get('input_output', '{}')
        input_output_data = json.loads(input_output_str)
        inputs = input_output_data.get('inputs', [])
        outputs = input_output_data.get('outputs', [])
        num_tests = len(inputs)
        
        if num_tests == 0:
            num_tests = 1
            inputs = [[""]]
            outputs = [""]
            
    except (json.JSONDecodeError, TypeError, AttributeError):
        num_tests = 1
        inputs = [[""]]
        outputs = [""]
    
    if debug:
        print(f"Running {num_tests} test cases")
    
    # Execute the code for each test case
    results = []
    all_passed = True
    
    for i, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
        try:
            if debug:
                print(f"Testing case {i+1}: input={test_input}, expected={expected_output}")
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                # Capture stdout
                old_stdout = sys.stdout
                old_stdin = sys.stdin
                sys.stdout = captured_output = io.StringIO()
                
                # Prepare input
                if isinstance(test_input, list) and len(test_input) > 0:
                    input_str = str(test_input[0])
                else:
                    input_str = str(test_input) if test_input else ""
                
                sys.stdin = io.StringIO(input_str)
                
                # Execute the code
                exec_globals = {
                    '__name__': '__main__',
                    '__builtins__': __builtins__,
                }
                exec(test, exec_globals)
                
                # Get output
                actual_output = captured_output.getvalue().strip()
                expected_output_str = str(expected_output).strip()
                
                # Compare outputs
                test_passed = actual_output == expected_output_str
                results.append(test_passed)
                
                if debug:
                    print(f"Case {i+1} - Expected: '{expected_output_str}', Got: '{actual_output}', Passed: {test_passed}")
                
                if not test_passed:
                    all_passed = False
                    
            finally:
                # Restore stdout/stdin and disable alarm
                sys.stdout = old_stdout
                sys.stdin = old_stdin
                signal.alarm(0)
                
        except TimeoutException:
            if debug:
                print(f"Case {i+1} timed out")
            results.append(False)
            all_passed = False
            
        except Exception as e:
            if debug:
                print(f"Case {i+1} failed with exception: {e}")
            results.append(False)
            all_passed = False
    
    # Return execution results
    status = "passed" if all_passed else "partial" if any(results) else "failed"
    return results, {"status": status, "method": "execution", "num_tests_run": num_tests} 