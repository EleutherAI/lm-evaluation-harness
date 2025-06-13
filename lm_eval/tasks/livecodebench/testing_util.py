# lm_eval/tasks/livecodebench/testing_util.py
# Comprehensive testing utilities based on evalscope implementation
import ast
import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import pickle
import platform
import resource
import signal
import sys
import time
import zlib
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from functools import partial
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch
import logging

# Configure logger
logger = logging.getLogger(__name__)

import_string = 'from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n'

# Global flag to control output verbosity - set to False for cleaner logs  
VERBOSE_OUTPUT = False

def set_verbose_output(verbose: bool):
    """Set the verbosity level for test output logging."""
    global VERBOSE_OUTPUT
    VERBOSE_OUTPUT = verbose

def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[:length // 2] + '...(truncated) ...' + s[-length // 2:]


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# stuff for setting up signal timer
class TimeoutException(Exception):
    pass


def timeout_handler(debug, signum, frame):
    if debug:
        logger.info('timeout occured: alarm went off')
    raise TimeoutException


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + '\n' + ast.unparse(last_block.body)  # type: ignore
                )
    except:
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name='wrapped_function',
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string + '\n' + ast.unparse(import_stmts)  # type: ignore
            + '\n' + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception as e:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = '\n'.join(inputs)

    inputs_line_iterator = iter(inputs.split('\n'))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator))
    @patch('sys.stdin.readlines', lambda *args: inputs.split('\n'))
    @patch('sys.stdin.read', lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception as e:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType('tmp_sol', '')
        exec(code, tmp_sol.__dict__)
        if 'class Solution' in code:
            # leetcode wraps solutions in `Solution`
            # this is a hack to check if it is leetcode solution or not
            # currently livecodebench only supports LeetCode but
            # else condition allows future extensibility to other platforms
            compiled_sol = tmp_sol.Solution()
        else:
            # do nothing in the other case since function is accesible
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str) -> tuple[bool, list[Decimal]]:
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    ## you don't want empty lines to add empty list after splitlines!
    val = val.strip()

    return [val_line.strip() for val_line in val.split('\n')]


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    # call-based clean up logic
    # need to wrap in try-catch logic after to catch the correct errors, but for now this is fine.
    code = import_string + '\n\n' + code
    
    # Suppress stdout during compilation and execution
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        compiled_sol = compile_code(code, timeout)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None:
        return

    all_inputs = [[json.loads(line) for line in inputs.split('\n')] for inputs in all_inputs]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    passed_tests = 0
    total_tests = len(all_inputs)
    
    test_message = f"\n🧪 Testing function-based problem with {total_tests} test cases:"
    print(test_message)
    print(test_message, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        # faulthandler.enable()
        try:
            # can lock here so time is useful
            start = time.time()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            # ground truth sequences are not tuples
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out
            all_results.append(tmp_result)
            
            if tmp_result:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            
            if VERBOSE_OUTPUT or not tmp_result:
                # Only show detailed output in verbose mode or on failure
                test_result = f"Test {idx + 1}: {status}"
                input_info = f"Input: {truncatefn(str(gt_inp))}"
                output_info = f"Generated Output: {truncatefn(str(prediction))}"
                truth_info = f"Ground Truth: {truncatefn(str(gt_out))}"
                
                print(test_result)
                print(input_info)
                print(output_info)
                print(truth_info)
            else:
                # Summary mode - just show pass/fail without details
                print(f"Test {idx + 1}: {status}")
            
            sys.stdout.flush()

            if not tmp_result:
                final_result = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED"
                print(final_result)
                sys.stdout.flush()
                return all_results, {
                    'output': truncatefn(prediction),
                    'inputs': truncatefn(gt_inp),
                    'expected': truncatefn(gt_out),
                    'error_code': -2,
                    'error_message': 'Wrong Answer',
                }
        except Exception as e:
            signal.alarm(0)
            if 'timeoutexception' in repr(e).lower():
                all_results.append(-3)
                timeout_result = f"Test {idx + 1}: ⏰ TIMEOUT"
                timeout_input = f"Input: {truncatefn(str(gt_inp))}"
                timeout_error = f"Error: Time Limit Exceeded"
                timeout_final = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED (TIMEOUT)"
                
                print(timeout_result)
                print(timeout_input)
                print(timeout_error)
                print(timeout_final)
                sys.stdout.flush()
                
                return all_results, {
                    'error': repr(e),
                    'error_code': -3,
                    'error_message': 'Time Limit Exceeded',
                    'inputs': truncatefn(gt_inp),
                    'expected': truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                runtime_result = f"Test {idx + 1}: 💥 RUNTIME ERROR"
                runtime_input = f"Input: {truncatefn(str(gt_inp))}"
                runtime_error = f"Error: {repr(e)}"
                runtime_final = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED (RUNTIME ERROR)"
                
                print(runtime_result)
                print(runtime_input)
                print(runtime_error)
                print(runtime_final)
                sys.stdout.flush()
                
                return all_results, {
                    'error': repr(e),
                    'error_code': -4,
                    'error_message': 'Runtime Error',
                    'inputs': truncatefn(gt_inp),
                    'expected': truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            # faulthandler.disable()

    final_result = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - {'PASSED' if passed_tests == total_tests else 'FAILED'}"
    print(final_result)
    sys.stdout.flush()
    return all_results, {'execution time': total_execution}


def grade_stdio(
    code: str,
    all_inputs: list,
    all_outputs: list,
    timeout: int,
):
    ## runtime doesn't interact well with __name__ == '__main__'
    code = clean_if_name(code)

    ## we wrap the given code inside another function
    code = make_function(code)

    # Suppress stdout during compilation
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, 'wrapped_function')

    if method is None:
        return

    all_results = []
    total_execution_time = 0
    passed_tests = 0
    total_tests = len(all_inputs)
    
    test_message = f"\n🧪 Testing stdio-based problem with {total_tests} test cases:"
    print(test_message)
    sys.stdout.flush()
    
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        # faulthandler.enable()

        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if 'timeoutexception' in repr(e).lower():
                    all_results.append(-3)
                    timeout_result = f"Test {idx + 1}: ⏰ TIMEOUT"
                    timeout_input = f"Input: {truncatefn(str(gt_inp))}"
                    timeout_error = f"Error: Time Limit Exceeded"
                    timeout_final = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED (TIMEOUT)"
                    
                    print(timeout_result)
                    print(timeout_input)
                    print(timeout_error)
                    print(timeout_final)
                    sys.stdout.flush()
                    
                    return all_results, {
                        'error': repr(e),
                        'error_code': -3,
                        'error_message': 'Time Limit Exceeded',
                        'inputs': truncatefn(gt_inp),
                        'expected': truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    runtime_result = f"Test {idx + 1}: 💥 RUNTIME ERROR"
                    runtime_input = f"Input: {truncatefn(str(gt_inp))}"
                    runtime_error = f"Error: {repr(e)}"
                    runtime_final = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED (RUNTIME ERROR)"
                    
                    print(runtime_result)
                    print(runtime_input)
                    print(runtime_error)
                    print(runtime_final)
                    sys.stdout.flush()
                    
                    return all_results, {
                        'error': repr(e),
                        'error_code': -4,
                        'error_message': 'Runtime Error',
                        'inputs': truncatefn(gt_inp),
                        'expected': truncatefn(gt_out),
                    }

            finally:
                signal.alarm(0)
                # faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        ## WA happens in multiple circumstances
        ## so cache the return to make it clean!
        WA_send_args = {
            'output': truncatefn(prediction),
            'inputs': truncatefn(gt_inp),
            'expected': truncatefn(gt_out),
            'error_code': -2,
        }

        test_passed = True
        error_message = ""

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args['error_message'] = 'Wrong answer: mismatched output length'
            test_passed = False
            error_message = f"Output length mismatch: got {len(stripped_prediction_lines)} lines, expected {len(stripped_gt_out_lines)} lines"
        else:
            for output_line_idx, (
                    stripped_prediction_line,
                    stripped_gt_out_line,
            ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
                WA_send_args['error_message'] = (
                    f'Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}'
                )

                ## CASE 1: exact match
                if stripped_prediction_line == stripped_gt_out_line:
                    continue

                ## CASE 2: element-wise comparision
                ## if there are floating elements
                ## use `decimal` library for good floating point comparision
                ## otherwise gotcha: np.isclose(50000000000000000, 50000000000000001) = True
                ## note that we should always be able to convert to decimals

                success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
                if not success:
                    all_results.append(-2)
                    test_passed = False
                    error_message = f"Line {output_line_idx}: '{stripped_prediction_line}' != '{stripped_gt_out_line}'"
                    break
                success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
                if not success:
                    all_results.append(-2)
                    test_passed = False
                    error_message = f"Line {output_line_idx}: '{stripped_prediction_line}' != '{stripped_gt_out_line}'"
                    break

                if decimal_prediction_line == decimal_gtout_line:
                    continue

                all_results.append(-2)
                test_passed = False
                error_message = f"Line {output_line_idx}: '{stripped_prediction_line}' != '{stripped_gt_out_line}'"
                break

        if test_passed:
            all_results.append(True)
            passed_tests += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        if VERBOSE_OUTPUT or not test_passed:
            # Only show detailed output in verbose mode or on failure
            test_result = f"Test {idx + 1}: {status}"
            input_info = f"Input: {truncatefn(str(gt_inp))}"
            output_info = f"Generated Output: {truncatefn(prediction)}"
            truth_info = f"Ground Truth: {truncatefn(gt_out)}"
            
            print(test_result)
            print(input_info)
            print(output_info)
            print(truth_info)
            
            if not test_passed:
                error_info = f"Error: {error_message}"
                print(error_info)
        else:
            # Summary mode - just show pass/fail without details
            print(f"Test {idx + 1}: {status}")
        
        sys.stdout.flush()

        if not test_passed:
            final_result = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - FAILED"
            print(final_result)
            sys.stdout.flush()
            return all_results, WA_send_args

    final_result = f"\n📊 Problem Result: {passed_tests}/{total_tests} tests passed - {'PASSED' if passed_tests == total_tests else 'FAILED'}"
    print(final_result)
    sys.stdout.flush()
    return all_results, {'execution time': total_execution_time}


def run_test(sample, test=None, debug=False, timeout=6):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    timeout_handler_wrapper = partial(timeout_handler, debug)
    signal.signal(signal.SIGALRM, timeout_handler_wrapper)

    # Disable functionalities that can make destructive changes to the test.
    # max memory is set to 4GB
    reliability_guard()

    if debug:
        logger.info(f'start = {datetime.now().time()}')

    try:
        in_outs = json.loads(sample['input_output'])
    except ValueError as e:
        raise e
        in_outs = None

    if in_outs:
        if in_outs.get('fn_name') is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None

        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = in_outs['fn_name']

    if debug:
        logger.info(f'loaded input_output = {datetime.now().time()}')

    if test is None:
        assert False, 'should not happen: test code is none'
        return in_outs, {'error': 'no test code provided'}
    elif test is not None:
        results = []
        sol = import_string
        if debug:
            logger.info(f'loading test code = {datetime.now().time()}')

        if which_type == CODE_TYPE.call_based:
            signal.alarm(timeout)
            try:
                results, metadata = grade_call_based(
                    code=test,
                    all_inputs=in_outs['inputs'],
                    all_outputs=in_outs['outputs'],
                    fn_name=method_name,
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    'error_code': -4,
                    'error_message': f'Error during testing: {e}',
                }
            finally:
                signal.alarm(0)
        elif which_type == CODE_TYPE.standard_input:
            # sol
            # if code has if __name__ == "__main__": then remove it

            signal.alarm(timeout)
            try:
                results, metadata = grade_stdio(
                    code=test,
                    all_inputs=in_outs['inputs'],
                    all_outputs=in_outs['outputs'],
                    timeout=timeout,
                )
                return results, metadata
            except Exception as e:
                return [-4], {
                    'error_code': -4,
                    'error_message': f'Error during testing: {e}',
                }
            finally:
                signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    # faulthandler.disable()

    import builtins

    # builtins.exit = None
    builtins.quit = None

    import os

    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys

    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None 