from typing import Dict, List, Any
import re
import json
import multiprocessing
import numpy as np
from collections import defaultdict

# Configure logger (assuming lm_eval's logging infrastructure)
from lm_eval.logging import get_logger
logger = get_logger(__name__)

# --- Helper functions from LiveCodeBench context ---

def extract_code_generation(model_output: str, model_type: str = 'chat'):
    # modified from
    outputlines = model_output.split('\\n')
    # TODO: handle codellama

    if model_type == 'base':
        return model_output.strip()
    elif model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
    else:
        raise ValueError(f'Invalid mode type: {model_type}')

    if len(indexlines) < 2:
        return ''
    return '\\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])


def codegen_check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.

    The global timeout is to catch some extreme/rare cases not handled by the
    timeouts inside `run_test`
    """

    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        # TODO: The following import `run_test` relies on `testing_util.py`
        # to be present in the same directory (lm_eval/tasks/livecodebench/).
        from .testing_util import run_test
        res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
        result.append(res)
        metadata_list.append(metadata)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    # Calculate global_timeout based on the number of test cases in the sample
    try:
        input_output_data = json.loads(sample['input_output'])
        num_tests = len(input_output_data.get('inputs', []))
        if num_tests == 0: # Fallback if 'inputs' is empty or not found
            num_tests = 1 
    except (json.JSONDecodeError, TypeError, KeyError):
        num_tests = 1 # Default to 1 test case if 'input_output' is not valid JSON or structure is unexpected

    global_timeout = (timeout + 1) * num_tests

    if debug:
        logger.info(f'global timeout = {global_timeout}')
    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
    if not result:
        # Ensure result is populated even on timeout to avoid IndexError
        result.append([-1] * num_tests) # Mark all tests as failed on global timeout
        if debug:
            logger.info('global timeout occured: alarm went off')
    
    # Ensure metadata_list is populated
    if not metadata_list:
        metadata_list.append({})

    return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list, debug: bool, timeout: int):
    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        curr_metadata = {} # Initialize curr_metadata
        try:
            curr_res, curr_metadata = codegen_check_correctness(sample, o, timeout=timeout, debug=debug)
            if debug:
                logger.info(f'\\nSuccessful compilation of task {o_idx}!')
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
                    logger.info(f'Results were not True for all test cases'
                                f' {curr_res=}\\n')
        except Exception as e:
            if debug:
                logger.info(f'Compilation failed, test framework exception'
                            f' = {repr(e)}{e}\\n')
            # Ensure curr_res has a default error value if an exception occurs early
            if curr_res == [-2]: # only if not already set by codegen_check_correctness
                 try:
                    input_output_data = json.loads(sample['input_output'])
                    num_tests = len(input_output_data.get('inputs', []))
                    if num_tests == 0: num_tests = 1
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
            logger.info(f'Sample\\n{r}\\nResult\\n{res[i]}')
            logger.info('*' * 30 + '\\n\\n')
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 1, # Changed default to 1 for lm-eval context
    timeout=6,
):
    results = {}
    metadata = {}

    for index in range(len(generations_list)):
        problem_generations = generations_list[index]
        sample = samples_list[index]

        # Note: The original code might have intended to use num_process_evaluate for parallel processing here.
        # For simplicity and lm-eval integration, this is currently sequential.
        # If parallel processing per problem is needed, Pool could be used here.
        result, meta = evaluate_generations_by_problem(problem_generations, sample, debug, timeout)
        results[index] = result
        metadata[index] = meta

    assert len(results) == len(
        generations_list), f'results = {len(results)} inputs = {len(generations_list)} {results=}'

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1], # Defaulted to [1] for pass@1
    num_process_evaluate=1, # Changed default to 1
    timeout=6,
    debug=False,
):
    # TODO: The following import `compute_metrics_from_results` relies on `pass_k_utils.py`
    # to be present in the same directory (lm_eval/tasks/livecodebench/).
    from .pass_k_utils import compute_metrics_from_results

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
            generations_linear.append([generation]) # evaluate_generations expects a list of generations for a sample
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
        # sub_results is a list of results for each generation attempt (here, only one attempt)
        # and each result itself is a list of booleans/ints for test cases.
        results[original_idx].append(sub_results[0])


    for idx_linear, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        original_idx = remap_index[idx_linear]
        metadatas[original_idx].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata_payload = []
    for key in sorted(list(metadatas.keys())):
        # Ensure metadata for each generation is handled correctly
        # If generations_list[key] has multiple generations, metadatas[key] should too.
        # For current setup (pass@1, one generation per sample in this call), this is simpler.
        current_metadata_list = metadatas[key]
        final_metadata_payload.append([json.dumps(x) for x in current_metadata_list])

    return metrics, results, final_metadata_payload


# --- lm-eval task-specific functions ---

def doc_to_target(doc: dict) -> dict:
    """
    Returns the full document, as evaluation functions require it.
    """
    return doc

def postprocess_generation(model_output: str) -> str:
    """
    Extracts the generated code from the model's output.
    """
    return extract_code_generation(model_output)

def process_results(doc: dict, results: List[str]) -> Dict[str, float]:
    """
    Processes the results for a single document and calculates pass@1.

    :param doc: The document dictionary.
    :param results: A list of model generations (typically one for pass@1).
    :return: A dictionary with the pass@1 metric.
    """
    if not results:
        return {"pass@1": 0.0}

    # We typically evaluate the first generation for pass@1
    generated_code = postprocess_generation(results[0])

    # The `codegen_metrics` function expects lists of samples and generations
    samples_list = [doc]
    # `generations_list` is a list of lists, where each inner list contains k generations for a sample.
    # For pass@1, k=1, so it's a list containing one list which contains the single generation.
    generations_list = [[generated_code]]

    # Parameters for codegen_metrics:
    # k_list=[1] for pass@1
    # num_process_evaluate=1 (sequential processing for simplicity in lm-eval)
    # timeout and debug can be configured if needed, using defaults for now.
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
        # The adapter divides by 100. Assuming `pass@1` from `compute_metrics_from_results` is in percentage.
        pass_at_1 = metrics_dict.get("pass@1", 0.0) / 100.0
    except Exception as e:
        logger.error(f"Error during livecodebench metric calculation for doc_id {doc.get('id', 'unknown')}: {e}")
        pass_at_1 = 0.0
        # Optionally, include more details in the output or log extensively
        # For example, return an error metric: return {"pass@1": 0.0, "error": 1.0}

    return {"pass@1": pass_at_1}