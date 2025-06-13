from typing import Dict, List, Any
import re
import json
import base64
import zlib
import multiprocessing
import numpy as np
from collections import defaultdict
import logging
import pickle
import sys

# Configure logger
logger = logging.getLogger(__name__)

# Global counters for tracking evaluation progress
_global_problem_counter = 0
_global_problems_passed = 0
_global_total_problems = 0

def reset_global_counters():
    """Reset global counters for a new evaluation run."""
    global _global_problem_counter, _global_problems_passed, _global_total_problems
    _global_problem_counter = 0
    _global_problems_passed = 0
    _global_total_problems = 0

def update_global_counters(passed: bool, total_problems: int = None):
    """Update global counters with results from a single problem."""
    global _global_problem_counter, _global_problems_passed, _global_total_problems
    _global_problem_counter += 1
    if passed:
        _global_problems_passed += 1
    if total_problems is not None:
        _global_total_problems = total_problems

def print_final_accuracy():
    """Print the final accuracy statistics."""
    global _global_problem_counter, _global_problems_passed, _global_total_problems
    if _global_problem_counter > 0:
        accuracy = (_global_problems_passed / _global_problem_counter) * 100
        print(f"\n" + "="*60)
        print(f"🎯 FINAL EVALUATION RESULTS")
        print(f"="*60)
        print(f"Total Problems Evaluated: {_global_problem_counter}")
        print(f"Problems Passed: {_global_problems_passed}")
        print(f"Problems Failed: {_global_problem_counter - _global_problems_passed}")
        print(f"Final Accuracy: {accuracy:.2f}% ({_global_problems_passed}/{_global_problem_counter})")
        print(f"="*60)

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
        
        # If no code blocks found, return empty string
        return ''
    else:
        raise ValueError(f'Invalid model type: {model_type}')


def codegen_check_correctness(sample, generation, timeout, debug=False):
    """Check correctness of code generation with a global timeout."""
    
    def _temp_run(sample, generation, debug, result, metadata_list, timeout):
        try:
            from lm_eval.tasks.livecodebench.testing_util import run_test
        except ImportError:
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
    global_timeout = (timeout + 1) * len(json.loads(sample['input_output'])['inputs'])
    if debug:
        logger.info(f'global timeout = {global_timeout}')
    p.join(timeout=global_timeout)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample['input_output'])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs['inputs']))]]
        if debug:
            logger.info('global timeout occured: alarm went off')
    return result[0], metadata_list[0]


def evaluate_generations_by_problem(problem_generations: list, sample: list, debug: bool, timeout: int):
    """Evaluate each problem.

    Args:
        problem_generations:
        sample:
        debug:
        timeout
    """
    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
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
            # break
            curr_metadata = {}
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
    num_process_evaluate: int = 16,  # This parameter will be unused
    timeout=6,
):
    """We take the list of code generations and try to compile them and the run
    their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS
            dataset)
        level: difficulty level used in the generation, can be "all",
            "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is
            a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test
            case [True] = passed test case
    """
    results = {}
    metadata = {}

    for index in range(len(generations_list)):
        problem_generations = generations_list[index]
        sample = samples_list[index]

        result, meta = evaluate_generations_by_problem(problem_generations, sample, debug, timeout)
        results[index] = result
        metadata[index] = meta

    assert len(results) == len(
        generations_list), f'results = {len(results)} inputs = {len(generations_list)} {results=}'

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
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
        assert isinstance(generation_list, list), generations_list[0]
        for generation in generation_list:
            assert isinstance(generation, str), generations_list[0]
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

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(), key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        assert len(final_metadata[i]) == len(generations_list[0]), f'{len(final_metadata[i])=}'

    return [metrics, results, final_metadata]


def transform_data_item(item):
    """Transform a single data item to match evalscope format."""
    # Define the format prompt constants
    FORMATTING_MESSAGE_WITH_STARTER_CODE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'
    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'
    
    # starter_code
    if item.get('starter_code'):
        format_prompt = f'### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'
        format_prompt += f"```python\n{item['starter_code']}\n```\n\n"
    else:
        format_prompt = f'### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n'
        format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'

    item['format_prompt'] = format_prompt

    # load test cases
    public_test_cases = item.get('public_test_cases', '[]')
    try:
        public_test_cases = json.loads(public_test_cases)
    except:
        public_test_cases = []

    private_test_cases = item.get('private_test_cases', '[]')
    try:
        private_test_cases = json.loads(private_test_cases)
    except Exception:
        try:
            # Handle compressed/pickled private test cases
            private_test_cases = json.loads(
                pickle.loads(zlib.decompress(base64.b64decode(private_test_cases.encode('utf-8')))))
        except Exception:
            private_test_cases = []

    # load metadata
    metadata = item.get('metadata', '{}')
    try:
        metadata = json.loads(metadata)
    except:
        metadata = {}
    
    evaluation_sample = json.dumps({
        'inputs': [t['input'] for t in public_test_cases + private_test_cases],
        'outputs': [t['output'] for t in public_test_cases + private_test_cases],
        'fn_name': metadata.get('func_name', None),
    })
    item['evaluation_sample'] = evaluation_sample

    return item


# --- lm-eval task-specific functions ---

def doc_to_text_with_format(doc: dict) -> str:
    """
    Generate the full prompt text including the format prompt.
    This function creates the complete prompt that will be sent to the model.
    Uses the exact format from evalscope implementation.
    """
    # System prompt (this would typically be handled by the model's system message)
    system_prompt = 'You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.'
    
    # Generate format prompt based on starter_code
    FORMATTING_MESSAGE_WITH_STARTER_CODE = 'You will use the following starter code to write the solution to the problem and enclose your code within delimiters.'
    FORMATTING_WITHOUT_STARTER_CODE = 'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.'
    
    if doc.get('starter_code'):
        format_prompt = f'### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n'
        format_prompt += f"```python\n{doc['starter_code']}\n```\n\n"
    else:
        format_prompt = f'### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n'
        format_prompt += '```python\n# YOUR CODE HERE\n```\n\n'
    
    # Use the exact prompt template format
    prompt_template = '### Question:\n{question_content}\n\n{format_prompt} ### Answer: (use the provided format with backticks)\n\n'
    
    # Format the template with the actual content
    full_prompt = prompt_template.format(
        question_content=doc['question_content'],
        format_prompt=format_prompt
    )
    
    return full_prompt


def doc_to_target(doc: dict) -> dict:
    """
    Returns the document with properly formatted input_output field.
    Uses the same transformation logic as evalscope.
    """
    # Make a copy to avoid modifying the original
    processed_doc = doc.copy()
    
    # Transform the document using evalscope logic
    transformed_doc = transform_data_item(processed_doc)
    
    # The evaluation_sample field becomes the input_output field
    transformed_doc['input_output'] = transformed_doc['evaluation_sample']
    
    return transformed_doc


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
    global _global_problem_counter, _global_problems_passed, _global_total_problems
    
    # Reset counters on first problem
    if _global_problem_counter == 0:
        reset_global_counters()
        message = f"\n🚀 Starting LiveCodeBench Evaluation..."
        print(message)
        sys.stdout.flush()
    
    if not results:
        update_global_counters(passed=False)
        return {"acc": 0.0}

    # We typically evaluate the first generation for pass@1
    generated_code = postprocess_generation(results[0])

    # Transform the document to ensure input_output field exists
    transformed_doc = transform_data_item(doc.copy())
    transformed_doc['input_output'] = transformed_doc['evaluation_sample']

    # The `codegen_metrics` function expects lists of samples and generations
    samples_list = [transformed_doc]
    generations_list = [[generated_code]]

    timeout = 6 
    debug = False

    try:
        # Create detailed question header
        question_id = doc.get('id', 'unknown')
        question_content = doc.get('question_content', '')
        question_preview = question_content[:200] + ('...' if len(question_content) > 200 else '')
        
        eval_header = f"\n{'='*80}\n🔍 LIVECODEBENCH EVALUATION - Processing Question {_global_problem_counter + 1}\n{'='*80}"
        question_id_info = f"📝 Question ID: {question_id}"
        question_content_info = f"📄 Question Content (first 200 chars): {question_preview}"
        
        print(eval_header)
        print(question_id_info)
        print(question_content_info)
        sys.stdout.flush()
        
        metrics, eval_results, final_metadata = codegen_metrics(
            samples_list=samples_list,
            generations_list=generations_list,
            k_list=[1],
            num_process_evaluate=1,
            timeout=timeout,
            debug=debug,
        )
        # Extract pass@1 and convert from percentage to decimal
        pass_at_1 = metrics.get("pass@1", 0.0)
        accuracy = pass_at_1 / 100.0
        
        # Update global counters
        problem_passed = accuracy > 0.0
        update_global_counters(passed=problem_passed)
        
        # Print progress update with separator
        current_accuracy = (_global_problems_passed / _global_problem_counter) * 100
        progress_message = f"📈 Progress: {_global_problem_counter} problems evaluated, {_global_problems_passed} passed ({current_accuracy:.1f}% overall)"
        closing_separator = f"{'='*80}"
        
        print(progress_message)
        print(closing_separator)
        sys.stdout.flush()
        
        logger.debug(f"Pass@1: {pass_at_1}, Accuracy: {accuracy}")
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error during livecodebench metric calculation for doc_id {doc.get('id', 'unknown')}: {e}")
        logger.error(f"Full traceback: {error_traceback}")
        accuracy = 0.0
        
        # Update global counters for failed evaluation
        update_global_counters(passed=False)

    return {"acc": accuracy}

def configure_livecodebench_logging(verbose=False):
    """Configure LiveCodeBench logging verbosity.
    
    Args:
        verbose (bool): If True, shows detailed test inputs/outputs for all tests.
                       If False, only shows summary and details for failed tests.
    """
    try:
        from lm_eval.tasks.livecodebench.testing_util import set_verbose_output
    except ImportError:
        try:
            from .testing_util import set_verbose_output
        except ImportError:
            # Testing util may not be available in all contexts
            pass
    else:
        set_verbose_output(verbose)

# Configure default logging to be non-verbose (cleaner output)
configure_livecodebench_logging(verbose=False)