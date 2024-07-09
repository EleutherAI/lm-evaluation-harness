import os
from dataclasses import asdict
from typing import Dict, Union

import pytest

from lm_eval import evaluator
from lm_eval.evaluator import request_caching_arg_to_dict
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import TaskManager
from lm_eval.utils import (
    hash_string,
    simple_parse_args_string,
)

from .utils import ParseConfig, filter_dict, load_all_configs


def update_results(results: Dict, evaluation_tracker: EvaluationTracker) -> Dict:
    """
    Calculates hashes and obtains data from the evaluation tracker to update the results dictionary.
    This code was taken from the `lm_eval.loggers.evaluation_tracker` module.
    """
    # build results dict from save function of the evaluation tracker
    # we don't want to call it, as it will save the results to json files
    samples = results.get("samples")
    evaluation_tracker.general_config_tracker.log_end_time()
    task_hashes = {}
    if samples:
        for task_name, task_samples in samples.items():
            sample_hashes = [
                s["doc_hash"] + s["prompt_hash"] + s["target_hash"]
                for s in task_samples
            ]
            task_hashes[task_name] = hash_string("".join(sample_hashes))
    # update initial results dict
    results.update({"task_hashes": task_hashes})
    results.update(asdict(evaluation_tracker.general_config_tracker))
    return results


def compare_results(
    target: Union[ParseConfig, Dict],
    observed: Dict,
    config_name: str,
    module_name: str,
    recursive: bool = False,
):
    """
    Compares values in the target and observed ParseConfig/dictionaries, checking for equality or approximate equality for floats.
    Complex nested structures (e.g., dictionaries or lists) are compared only if recursive is set to True.

    Args:
        target: target ParseConfig or dictionary to compare with
        observed: observed dictionary to compare
        config_name: name of the config with the results to display in the error message
        module_name: associated name to the checked subresults to display in the error message
        recursive: whether to compare nested structures

    Example:
        target = {
            "transformers_version": "4.41.1",
            "system_instruction_sha": None,
            "versions": {"drop": 3.0}  # Nested structure
        }
        observed = {
            "transformers_version": "4.41.1",
            "system_instruction_sha": None,
            "versions": {"drop": 3.0}  # Nested structure
        }

        The function will compare 'transformers_version' and 'system_instruction_sha' but will skip 'versions'
        as it is a nested structure.
    """
    if not target:
        raise ValueError("Target results are empty.")
    for key in target.keys():
        target_val = target[key]
        # compare nested objects only if recursive is set to True
        if isinstance(target_val, ParseConfig):
            if recursive:
                compare_results(target_val, observed[key], config_name, key, recursive)
                continue
            else:
                continue
        if key not in observed:
            raise ValueError(
                f"Config: '{config_name} - {module_name}' failed. {key}: "
                f"Expected: {repr(target_val)}, got: None"
            )
        observed_val = observed[key]
        if isinstance(target_val, float):
            assert target_val == pytest.approx(observed_val, abs=1e-4), (
                f"Config: '{config_name} - {module_name}' failed. {key}: "
                f"Expected: {repr(target_val)}, got: {repr(observed_val)}"
            )
        else:
            assert target_val == observed_val, (
                f"Config: '{config_name} - {module_name}' failed. {key}: "
                f"Expected: {repr(target_val)}, got: {repr(observed_val)}"
            )


@pytest.fixture(scope="module", params=load_all_configs(os.getenv("TESTS_DEVICE")))
def load_config(request):
    """
    Loads all configs with the expected results for the tests.
    Config are chosen based on the device specified in the environment variable.
    If the device is not specified, the default device is "cpu".
    """
    return request.param


@pytest.fixture(scope="module")
def evaluation_results(load_config):
    """
    Runs evaluations for all loaded configs, returning the config and the results.
    """
    config = ParseConfig(load_config)

    evaluation_tracker_args = simple_parse_args_string(
        f"output_path={config.params.output_path}"
    )
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    task_manager = TaskManager(config.params.verbosity, include_path=None)
    seed = [0, 1234, 1234, 1234]
    request_caching_args = request_caching_arg_to_dict(cache_requests=None)

    all_results = {}
    for task_name, task in config.tasks.items():
        results = evaluator.simple_evaluate(
            model=config.params.model,
            model_args=config.params.model_args,
            tasks=[task_name],
            num_fewshot=task.num_fewshot,
            batch_size=config.params.batch_size,
            max_batch_size=config.params.max_batch_size,
            device=config.params.device,
            use_cache=config.params.use_cache,
            limit=task.limit,  # limit is varying depending on the task
            check_integrity=config.params.check_integrity,
            write_out=config.params.write_out,
            log_samples=config.params.log_samples,
            evaluation_tracker=evaluation_tracker,
            system_instruction=config.params.system_instruction,
            apply_chat_template=config.params.apply_chat_template,
            fewshot_as_multiturn=config.params.fewshot_as_multiturn,
            gen_kwargs=config.params.gen_kwargs,
            task_manager=task_manager,
            verbosity=config.params.verbosity,
            predict_only=config.params.predict_only,
            random_seed=seed[0],
            numpy_random_seed=seed[1],
            torch_random_seed=seed[2],
            fewshot_random_seed=seed[3],
            **request_caching_args,
        )
        results = update_results(results, evaluation_tracker)
        all_results[task_name] = results

    return config, all_results


def test_general_output(evaluation_results: Dict):
    """
    Compares the lowest level of the results dictionary.
    """
    config, all_results = evaluation_results
    for task_name, task in config.tasks.items():
        results = all_results[task_name]
        compare_results(
            config, results, config.params.name, "general_output", recursive=False
        )


def test_evaluation_config(evaluation_results: Dict):
    """
    Compares returned evaluation config.
    """
    config, all_results = evaluation_results
    for task_name, task in config.tasks.items():
        results = all_results[task_name]
        # filter config params used for setup but not present in the results
        expected_config_dict = config.params.to_dict().copy()
        config_name = expected_config_dict.pop("name")
        config_params_not_in_results = [
            "output_path",
            "system_instruction",
            "chat_template",
            "max_batch_size",
            "write_out",
            "check_integrity",
            "log_samples",
            "apply_chat_template",
            "fewshot_as_multiturn",
            "verbosity",
            "predict_only",
            "default_seed_string",
        ]
        for key in config_params_not_in_results:
            expected_config_dict.pop(key, None)
        compare_results(
            expected_config_dict,
            results["config"],
            config_name,
            "config",
            recursive=False,
        )


def test_tasks_configs(evaluation_results: Dict):
    """
    Compares configs for each task and subtask.
    """
    config, all_results = evaluation_results
    for task_name, task in config.tasks.items():
        results = all_results[task_name]
        # if configs has more than one key - process as multitask
        multitask = len(results["configs"].keys()) > 1
        if multitask:
            # filter subtasks for the current task
            subtasks = filter_dict(task.to_dict(), task_name)
        else:
            # exclude keys not present in evaluation results
            expected_task_dict = task.to_dict().copy()
            expected_task_dict.pop("limit")
            subtasks = {task_name: expected_task_dict}
        for subtask_name, subtask in subtasks.items():
            compare_results(
                subtask,
                results["configs"][subtask_name],
                config.params.name,
                subtask_name,
                recursive=False,
            )


def test_tasks_results(evaluation_results: Dict):
    """
    Compares results for each task and subtask. Subtasks are compared recursively.
    """
    config, all_results = evaluation_results
    for task_name, task_results in config.results.items():
        results = all_results[task_name]
        multitask = len(results["results"].keys()) > 1
        if multitask:
            results = results["results"]
        else:
            results = results["results"][task_name]
        compare_results(
            task_results,
            results,
            config.params.name,
            task_name,
            recursive=True,
        )


def test_tasks_n_samples(evaluation_results: Dict):
    """
    Compares n_samples for each task and subtask. Subtasks are compared recursively.
    """
    config, all_results = evaluation_results
    for task_name, task_n_samples in config.n_samples.items():
        results = all_results[task_name]
        multitask = len(results["n-samples"].keys()) > 1
        if multitask:
            results = results["n-samples"]
        else:
            results = results["n-samples"][task_name]
        compare_results(
            task_n_samples,
            results,
            config.params.name,
            task_name,
            recursive=True,
        )


def test_tasks_hashes(evaluation_results: Dict):
    """
    Compares task hashes for each task and subtask. Subtasks are compared recursively.
    """
    config, all_results = evaluation_results
    for task_name, task_hashes in config.task_hashes.items():
        results = all_results[task_name]
        multitask = len(results["configs"].keys()) > 1
        if multitask:
            compare_results(
                task_hashes,
                results["task_hashes"],
                config.params.name,
                task_name,
                recursive=True,
            )
        else:
            observed_task_hash = results["task_hashes"][task_name]
            assert task_hashes == observed_task_hash, (
                f"Config: '{config.params.name}' failed. {task_name}: "
                f"Expected: {repr(task_hashes)}, got: {repr(observed_task_hash)}"
            )


def test_tasks_versions(evaluation_results: Dict):
    """
    Compares versions for each task and subtask. Subtasks are compared recursively.
    """
    config, all_results = evaluation_results
    for task_name, task_versions in config.versions.items():
        results = all_results[task_name]
        multitask = len(results["versions"].keys()) > 1
        if multitask:
            compare_results(
                task_versions,
                results["versions"],
                config.params.name,
                task_name,
                recursive=True,
            )
        else:
            observed_version = results["versions"][task_name]
            assert task_versions == observed_version, (
                f"Config: '{config.params.name}' failed. {task_name}: "
                f"Expected: {repr(task_versions)}, got: {repr(observed_version)}"
            )
