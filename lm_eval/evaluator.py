import collections
import itertools
import logging
import random
from typing import Optional, Union

import numpy as np
import torch

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import (
    consolidate_results,
    eval_logger,
    get_git_commit_hash,
    get_sample_size,
    get_task_list,
    positional_deprecated,
    print_tasks,
    print_writeout,
    run_task_tests,
    simple_parse_args_string,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[str] = None,
    tasks=None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
    task_manager: TaskManager = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if use_cache is not None:
        eval_logger.info(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    eval_logger.info(
        "get_task_dict has been updated to accept an optional argument, `task_manager`"
        "Read more here:https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage"
    )
    task_dict = get_task_dict(tasks, task_manager)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

            if predict_only:
                log_samples = True
                eval_logger.info(
                    f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                )
                # we have to change the class properties post-hoc. This is pretty hacky.
                task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit: Optional[int] = None,
    bootstrap_iters: Optional[int] = 100000,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    # decontaminate = decontamination_ngrams_path is not None

    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)

    # get lists of group hierarchy and each type of request
    task_hierarchy, eval_tasks = get_task_list(task_dict)
    if not log_samples:
        assert all(
            "bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys()
            for task_output in eval_tasks
        ), "log_samples must be True for 'bypass' only tasks"
    for task_output in eval_tasks:
        task = task_output.task
        limit = get_sample_size(task, limit)
        task.build_all_requests(rank=lm.rank, limit=limit, world_size=lm.world_size)
        eval_logger.debug(
            f"Task: {task_output.task_name}; number of requests on this rank: {len(task.instances)}"
        )

        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                    }
                    example.update(metrics)
                    task_output.log_samples.append(example)
                for metric, value in metrics.items():
                    task_output.samples_metrics[(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.log_samples, object_gather_list=full_samples, dst=0
                )

                if RANK == 0:
                    task_output.log_samples = list(
                        itertools.chain.from_iterable(full_samples)
                    )

            # then collect metrics across all ranks
            # vals_torch = collections.defaultdict(list)
            for metrics in task_output.samples_metrics:
                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=task_output.samples_metrics[metrics],
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.samples_metrics[metrics] = list(
                        itertools.chain.from_iterable(metric_list)
                    )

    if RANK == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        results, samples, configs, versions, num_fewshot = consolidate_results(
            eval_tasks
        )

        ### Calculate group metrics ###
        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if len(task_list) == 0:
                    # task_hierarchy entries are either
                    # `group_name: [subtask1, subtask2, ...]`
                    # or `task_name: []`.
                    # we only want to operate on groups here.
                    continue
                metric_list = list(
                    {
                        key
                        for task in task_list
                        for key in results[task].keys()
                        if "_stderr" not in key and key not in ["alias", "samples"]
                    }
                )
                for metric in metric_list:
                    stderr = "_stderr,".join(metric.split(","))

                    # gather metrics, sizes, and stderrs from subtasks
                    metrics = [
                        results[task][metric]
                        for task in task_list
                        if metric in results[task]
                    ]  # TODO: copy?
                    stderrs = [
                        results[task][stderr]
                        for task in task_list
                        if stderr in results[task]
                    ]
                    sizes = [
                        results[task]["samples"]
                        for task in task_list
                        if metric in results[task]
                    ]

                    # compute group's pooled metric and stderr
                    results[group][
                        metric
                    ] = lm_eval.api.metrics.aggregate_subtask_metrics(metrics, sizes)
                    # TODO: calculate grouped metric using aggregation fn
                    if "N/A" in stderrs:
                        results[group][stderr] = "N/A"
                    else:
                        results[group][
                            stderr
                        ] = lm_eval.api.metrics.pooled_sample_stderr(stderrs, sizes)
                        # TODO: allow GroupConfigs to choose which variance formula is used, for back-compatibility
                        # To use the old (likely incorrect) variance formula, comment out the above and uncomment this line:
                        # results[group][stderr] = lm_eval.api.metrics.combined_sample_stderr(stderrs, sizes, metrics=metrics)

                    results[group]["samples"] = sum(sizes)

        results_agg = collections.defaultdict(dict)
        groups_agg = collections.defaultdict(dict)
        all_tasks_list = list(task_hierarchy.keys())
        left_tasks_list = []
        while True:
            add_tasks_list = list(k for k in results_agg.keys())
            left_tasks_list = sorted(list(set(all_tasks_list) - set(add_tasks_list)))
            if len(left_tasks_list) == 0:
                break

            _task_hierarchy = {
                k: v for k, v in task_hierarchy.items() if k in left_tasks_list
            }
            _results_agg, _groups_agg = print_tasks(_task_hierarchy, results)

            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

        for group_name, task_list in task_hierarchy.items():
            if task_list:
                num_fewshot[group_name] = num_fewshot[
                    task_list[0]
                ]  # TODO: validate this

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None
