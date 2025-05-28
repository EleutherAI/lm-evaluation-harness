# lm_eval/tasks/livecodebench/pass_k_utils.py
import numpy as np

def compute_metrics_from_results(results, k_list=[1]):
    """
    Compute pass@k metrics from evaluation results.
    
    Args:
        results: Dict where keys are problem indices and values are lists of lists of booleans/integers.
                Each inner list represents the outcome of test cases for one generation.
        k_list: List of k values to compute pass@k for (e.g., [1, 2, 5])
    
    Returns:
        Dict with pass@k metrics as percentages
    """
    total_problems = len(results)
    pass_at_k = {}

    for k in k_list:
        if total_problems == 0:
            pass_at_k[f"pass@{k}"] = 0.0
            continue

        num_passed = 0
        for problem_idx in results:
            # Get the first k generations for this problem
            problem_results_k_generations = results[problem_idx][:k]
            problem_passed = False
            for gen_result_list in problem_results_k_generations:
                # A generation is successful if all its test cases passed
                # Note: -1 represents error/timeout, -2 represents compilation failure
                # Only True (or 1) should be considered as passed
                if all(result == True or result == 1 for result in gen_result_list):
                    problem_passed = True
                    break
            if problem_passed:
                num_passed += 1
        
        pass_at_k[f"pass@{k}"] = (num_passed / total_problems) * 100.0 if total_problems > 0 else 0.0
        
    return pass_at_k

def postprocess_generation(model_output: str) -> str:
    """
    Extracts the generated code from the model's output.
    """
    return extract_code_generation(model_output) # model_type defaults to 'chat' 

def extract_code_generation(model_output: str, model_type: str = 'chat'):
    # modified from
    outputlines = model_output.split('\\n') # This is the version you want to keep
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