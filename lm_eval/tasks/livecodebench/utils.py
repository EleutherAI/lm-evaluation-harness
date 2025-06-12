from typing import Dict, List, Any
import re
import json
import base64
import zlib
import multiprocessing
import numpy as np
from collections import defaultdict
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Helper function for multiprocessing
def _temp_run_helper(sample, generation, debug, result, metadata_list, timeout):
    try:
        from lm_eval.tasks.livecodebench.testing_util import run_test
    except ImportError:
        from .testing_util import run_test
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def extract_code_generation(model_output: str, model_type: str = 'chat'):
    """Extract code from model output based on model type."""
    outputlines = model_output.split('\n')

    if model_type == 'base':
        return model_output.strip()
    elif model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
        
        # If we found code blocks, extract the code
        if len(indexlines) >= 2:
            return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])
        
        # If no code blocks found, check if the entire output looks like code
        # This handles cases where models generate raw Python code without markdown
        stripped_output = model_output.strip()
        if stripped_output:
            # Simple heuristic: if it contains common Python keywords, treat as code
            python_indicators = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'print(', 'return ']
            if any(indicator in stripped_output for indicator in python_indicators):
                return stripped_output
        
        # If no code blocks and doesn't look like code, return empty
        return ''
    else:
        raise ValueError(f'Invalid model type: {model_type}')


def codegen_check_correctness(sample, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout."""
    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run_helper,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    
    # Calculate global_timeout based on the number of test cases in the sample
    try:
        input_output_data = json.loads(sample['input_output'])
        num_tests = len(input_output_data.get('inputs', []))
        if num_tests == 0:
            num_tests = 1 
    except (json.JSONDecodeError, TypeError, KeyError):
        num_tests = 1

    global_timeout = (timeout + 1) * num_tests

    if debug:
        logger.info(f'global timeout = {global_timeout}')
    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
    if not result:
        # Ensure result is populated even on timeout to avoid IndexError
        result.append([-1] * num_tests)
        if debug:
            logger.info('global timeout occurred: alarm went off')
    
    # Ensure metadata_list is populated
    if not metadata_list:
        metadata_list.append({})

    return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list, debug: bool, timeout: int):
    """Evaluate multiple generations for a single problem."""
    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        curr_metadata = {}
        try:
            curr_res, curr_metadata = codegen_check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                logger.info(f'\nSuccessful compilation of task {o_idx}!')
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    logger.info(f'Results were not True for all test cases {curr_res=}\n')
        except Exception as e:
            if debug:
                logger.info(f'Compilation failed, test framework exception = {repr(e)}{e}\n')
            # Ensure curr_res has a default error value if an exception occurs early
            if curr_res == [-2]:
                 try:
                    input_output_data = json.loads(sample['input_output'])
                    num_tests = len(input_output_data.get('inputs', []))
                    if num_tests == 0: 
                        num_tests = 1
                 except:
                    num_tests = 1
                 curr_res = [-1] * num_tests

        finally:
            assert isinstance(curr_res, list)
            assert isinstance(curr_metadata, dict)
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            logger.info(f'Sample\n{r}\nResult\n{res[i]}')
            logger.info('*' * 30 + '\n\n')
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 1,
    timeout=6,
):
    """Evaluate generations for multiple problems."""
    results = {}
    metadata = {}

    for index in range(len(generations_list)):
        problem_generations = generations_list[index]
        sample = samples_list[index]

        result, meta = evaluate_generations_by_problem(problem_generations, sample, debug, timeout)
        results[index] = result
        metadata[index] = meta

    assert len(results) == len(generations_list), f'results = {len(results)} inputs = {len(generations_list)} {results=}'

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1],
    num_process_evaluate=1,
    timeout=6,
    debug=False,
):
    """Compute pass@k metrics for code generation evaluation."""
    try:
        from lm_eval.tasks.livecodebench.pass_k_utils import compute_metrics_from_results
    except ImportError:
        try:
            # Fallback for relative import
            from .pass_k_utils import compute_metrics_from_results
        except ImportError:
            # Last fallback: try to import from same directory
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from pass_k_utils import compute_metrics_from_results

    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample, generation_list) in enumerate(zip(samples_list, generations_list)):
        assert isinstance(generation_list, list), generation_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generation_list[0]
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx_linear, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        original_idx = remap_index[idx_linear]
        results[original_idx].append(sub_results[0])

    for idx_linear, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        original_idx = remap_index[idx_linear]
        metadatas[original_idx].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata_payload = []
    for key in sorted(list(metadatas.keys())):
        current_metadata_list = metadatas[key]
        final_metadata_payload.append([json.dumps(x) for x in current_metadata_list])

    return metrics, results, final_metadata_payload


# --- lm-eval task-specific functions ---

def doc_to_target(doc: dict) -> dict:
    """
    Returns the document with properly formatted input_output field.
    Converts public_test_cases and private_test_cases to the expected input_output format.
    ENSURES ALL PRIVATE TEST CASES ARE INCLUDED - NO SKIPPING ALLOWED.
    """
    # Make a copy to avoid modifying the original
    processed_doc = doc.copy()
    question_id = doc.get('question_id', 'unknown')
    
    # Check if input_output already exists (shouldn't happen with current dataset)
    if 'input_output' in processed_doc:
        return processed_doc
    
    # Convert test cases to input_output format
    inputs = []
    outputs = []
    
    # Process public test cases
    public_count = 0
    if 'public_test_cases' in processed_doc and processed_doc['public_test_cases']:
        try:
            public_test_cases = json.loads(processed_doc['public_test_cases'])
            for test_case in public_test_cases:
                if isinstance(test_case, dict) and 'input' in test_case and 'output' in test_case:
                    # Clean up the input/output strings (remove trailing newlines)
                    test_input = test_case['input'].rstrip('\n\r')
                    test_output = test_case['output'].rstrip('\n\r')
                    
                    # Format as expected by testing_util.py
                    inputs.append([test_input])
                    outputs.append(test_output)
                    public_count += 1
            logger.debug(f"Doc {question_id}: Successfully processed {public_count} public test cases")
        except Exception as e:
            logger.warning(f"Doc {question_id}: Failed to process public test cases: {e}")
    
    # Process private test cases - MUST SUCCEED OR FAIL DOCUMENT
    private_count = 0
    if 'private_test_cases' in processed_doc and processed_doc['private_test_cases']:
        private_encoded = processed_doc['private_test_cases'].strip()
        if private_encoded:
            private_test_cases = None
            decoding_error = None
            
            try:
                # Step 1: Base64 decode
                try:
                    private_decoded = base64.b64decode(private_encoded)
                except Exception as e:
                    raise ValueError(f"Base64 decoding failed: {e}")
                
                # Step 2: Decompress
                try:
                    private_decompressed = zlib.decompress(private_decoded)
                except Exception as e:
                    raise ValueError(f"Zlib decompression failed: {e}")
                
                # Step 3: Try multiple deserialization methods
                # Method 1: Pickle (most common)
                try:
                    import pickle
                    private_data = pickle.loads(private_decompressed)
                    # Handle nested JSON string in pickle
                    if isinstance(private_data, str):
                        private_test_cases = json.loads(private_data)
                    elif isinstance(private_data, (list, dict)):
                        private_test_cases = private_data
                    else:
                        raise ValueError(f"Unexpected pickle data type: {type(private_data)}")
                    logger.debug(f"Doc {question_id}: Successfully decoded with pickle")
                except Exception as pickle_error:
                    logger.debug(f"Doc {question_id}: Pickle decoding failed: {pickle_error}")
                    
                    # Method 2: Raw JSON with multiple encodings
                    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            decoded_text = private_decompressed.decode(encoding)
                            private_test_cases = json.loads(decoded_text)
                            logger.debug(f"Doc {question_id}: Successfully decoded with JSON ({encoding})")
                            break
                        except Exception as json_error:
                            logger.debug(f"Doc {question_id}: JSON decoding with {encoding} failed: {json_error}")
                            continue
                    
                    # Method 3: Try interpreting as direct binary data
                    if private_test_cases is None:
                        try:
                            # Sometimes the data might be double-compressed or have extra layers
                            try:
                                # Try double decompression
                                double_decompressed = zlib.decompress(private_decompressed)
                                private_test_cases = json.loads(double_decompressed.decode('utf-8'))
                                logger.debug(f"Doc {question_id}: Successfully decoded with double decompression")
                            except:
                                # Try as raw bytes
                                private_test_cases = json.loads(private_decompressed)
                                logger.debug(f"Doc {question_id}: Successfully decoded as raw bytes")
                        except Exception as raw_error:
                            logger.debug(f"Doc {question_id}: Raw decoding failed: {raw_error}")
                
                # Validate decoded data
                if private_test_cases is None:
                    raise ValueError("All decoding methods failed")
                
                if not isinstance(private_test_cases, list):
                    raise ValueError(f"Private test cases should be a list, got {type(private_test_cases)}")
                
                # Process the private test cases
                for i, test_case in enumerate(private_test_cases):
                    if isinstance(test_case, dict) and 'input' in test_case and 'output' in test_case:
                        test_input = test_case['input'].rstrip('\n\r')
                        test_output = test_case['output'].rstrip('\n\r')
                        
                        inputs.append([test_input])
                        outputs.append(test_output)
                        private_count += 1
                    else:
                        logger.warning(f"Doc {question_id}: Invalid private test case {i}: {test_case}")
                
                logger.info(f"Doc {question_id}: Successfully processed {private_count} private test cases")
                        
            except Exception as e:
                decoding_error = str(e)
                logger.error(f"Doc {question_id}: CRITICAL - Failed to decode private test cases: {e}")
                logger.error(f"Doc {question_id}: Private test case data length: {len(private_encoded)} chars")
                logger.error(f"Doc {question_id}: First 100 chars: {private_encoded[:100]}")
                
                # STRICT MODE: Do not allow documents with failed private test cases
                # This prevents inflated accuracy scores
                raise ValueError(f"Document {question_id} has undecoded private test cases - evaluation would be invalid")
    
    # Validate we have test cases
    total_cases = len(inputs)
    if total_cases == 0:
        logger.error(f"Doc {question_id}: CRITICAL - No valid test cases found (public: {public_count}, private: {private_count})")
        raise ValueError(f"Document {question_id} has no valid test cases - cannot evaluate")
    
    # Create the input_output field
    processed_doc['input_output'] = json.dumps({
        'inputs': inputs,
        'outputs': outputs
    })
    
    logger.info(f"Doc {question_id}: Successfully created input_output with {total_cases} test cases (public: {public_count}, private: {private_count})")
    return processed_doc

def postprocess_generation(model_output: str) -> str:
    """Extracts the generated code from the model's output."""
    return extract_code_generation(model_output)

def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Processes the results for a single document and calculates accuracy.

    :param doc: The document dictionary.
    :param results: A list of model generations (typically one for pass@1).
    :return: A dictionary with the accuracy metric.
    """
    if not results:
        return {"acc": 0.0}

    # We typically evaluate the first generation for pass@1
    generated_code = postprocess_generation(results[0])

    # The `codegen_metrics` function expects lists of samples and generations
    samples_list = [doc]
    generations_list = [[generated_code]]

    timeout = 6 
    debug = False

    try:
        metrics_dict, _, _ = codegen_metrics(
            samples_list=samples_list,
            generations_list=generations_list,
            k_list=[1],
            num_process_evaluate=1,
            timeout=timeout,
            debug=debug,
        )
        # DEBUG: Print the raw metrics to understand the issue
        logger.debug(f"Raw metrics_dict: {metrics_dict}")
        pass_at_1 = metrics_dict.get("pass@1", 0.0)
        logger.debug(f"pass@1 value: {pass_at_1}")
        
        # Convert from percentage to decimal and use 'acc' key to match YAML
        accuracy = pass_at_1 / 100.0
        logger.debug(f"Final accuracy after division: {accuracy}")
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error during livecodebench metric calculation for doc_id {doc.get('id', 'unknown')}: {e}")
        logger.error(f"Full traceback: {error_traceback}")
        accuracy = 0.0

    return {"acc": accuracy}