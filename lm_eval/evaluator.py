from __future__ import annotations

import itertools
import json
import logging
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

import lm_eval.api.model
import lm_eval.api.registry
from lm_eval.caching.cache import delete_cache
from lm_eval.defaults import DEFAULT_OTHER_SEED, DEFAULT_RANDOM_SEED, LMEVAL_HASHMM
from lm_eval.evaluator_utils import (
    ResultAcc,
    _handle_back_comp,
    _log_selected_tasks,
    _process_results,
    get_sample_size,
    print_writeout,
    run_task_tests,
)
from lm_eval.loggers.utils import add_env_info, add_tokenizer_info, get_git_commit_hash
from lm_eval.tasks import TaskManager
from lm_eval.utils import (
    handle_non_serializable,
    hash_dict_images,
    hash_string,
    positional_deprecated,
    set_torch_seed,
    setup_logging,
    simple_parse_args_string,
    wrap_text,
)


if TYPE_CHECKING:
    from lm_eval.api.group import Group
    from lm_eval.api.model import LM
    from lm_eval.api.task import Task
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.result_schema import EvalResults
    from lm_eval.tasks.manager import TaskDict

    _NestedDict = dict[Group, dict[str, Task] | Group] | dict[str, Task]

eval_logger = logging.getLogger(__name__)


@positional_deprecated
def simple_evaluate(
    model: str | LM,
    model_args: str | dict[str, str | int | float] | None = None,
    tasks: list[str | dict[str, Any] | Task] | None = None,
    num_fewshot: int | None = None,
    batch_size: int | str | None = None,
    max_batch_size: int | None = None,
    device: str | None = None,
    use_cache: str | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: int | float | None = None,
    samples: dict[str, list[int]] | None = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: EvaluationTracker | None = None,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = True,
    gen_kwargs: str | dict[str, str | float | int] | None = None,
    task_manager: TaskManager | None = None,
    verbosity=None,
    predict_only: bool = False,
    random_seed: int = DEFAULT_RANDOM_SEED,
    numpy_random_seed: int = DEFAULT_OTHER_SEED,
    torch_random_seed: int = DEFAULT_OTHER_SEED,
    fewshot_random_seed: int = DEFAULT_OTHER_SEED,
    confirm_run_unsafe_code: bool = False,
    metadata: dict[str, Any] | None = None,
) -> EvalResults | None:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        model (str | LM): Name of model or LM object. See
            lm_eval.models.__init__.py for available aliases.
        model_args: String or dict arguments for each model class, e.g.,
            "pretrained=EleutherAI/pythia-1.3B,revision=main" or {"pretrained": "EleutherAI/pythia-1.3B"}.
            Ignored if ``model`` argument is a LM object.
        tasks (list[str | dict | Task]): List of task names or Task objects.
            Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined
            and type(task).__name__ otherwise.
        num_fewshot (int): Number of examples in few-shot context.
        batch_size (int | str | None): Batch size for model.
        max_batch_size (int | None): Maximal batch size to try with automatic
            batch size detection.
        device (str | None): PyTorch device (e.g. "cpu" or "cuda:0") for running
            models.
        use_cache (str | None): A path to a sqlite db file for caching model
            responses. `None` if not caching.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests (inputs). `None` if not caching.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`. `None` if not desired.
        delete_requests_cache (bool): Deletes all the request cache if set to
            `True`. `None` if not desired.
        limit (int | float | None): Limit the number of examples per task (only
            use this for testing). If <1, limit is a percentage of the total
            number of examples.
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
            Incompatible with `limit`.
        bootstrap_iters (int): Number of iterations for bootstrap statistics, used
            when calculating stderrs. Set to 0 for no stderr calculations to be
            performed.
        check_integrity (bool): Whether to run the relevant part of the test suite
            for the tasks.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents for
            per-sample measurement and post-hoc analysis.
        evaluation_tracker (EvaluationTracker | None): Tracker for logging
            experiment configuration and results.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        gen_kwargs (dict | str | None): Arguments for model generation. Ignored
            for all tasks with loglikelihood output_type.
        task_manager (TaskManager | None): Task manager instance to use.
        verbosity (str | None): Verbosity level for logging.
        predict_only (bool): If True, only model outputs will be generated and
            returned. Metrics will not be evaluated.
        random_seed (int): Random seed for python's random module. If set to None,
            the seed will not be set.
        numpy_random_seed (int): Random seed for numpy. If set to None, the seed
            will not be set.
        torch_random_seed (int): Random seed for torch. If set to None, the seed
            will not be set.
        fewshot_random_seed (int): Random seed for fewshot sampler random generator.
            If set to None, the seed of generator will be set to None.
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe (e.g. code execution tasks).
        metadata (dict | None): Additional metadata to be added to the task
            manager. Will get passed to the download function of the task.

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """
    if verbosity is not None:
        eval_logger.info("Setting verbosity through simple_evaluate is deprecated.")
    start_date = time.time()

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )

    _NEEDS_CHAT_TEMPLATE = ("inst", "chat")
    if (
        (
            isinstance(model_args, str)
            and any(kw in model_args.lower() for kw in _NEEDS_CHAT_TEMPLATE)
        )
        or (
            isinstance(model_args, dict)
            and any(
                any(kw in str(v).lower() for kw in _NEEDS_CHAT_TEMPLATE)
                for v in model_args.values()
            )
        )
    ) and not apply_chat_template:
        eval_logger.warning(
            wrap_text(
                f"""pretrained={model_args.get("pretrained") if isinstance(model_args, dict) else model_args} appears to be an
                instruct or chat variant but chat template is not applied.
                Recommend setting `apply_chat_template` (optionally `fewshot_as_multiturn`).""",
            )
        )

    if delete_requests_cache:
        eval_logger.info("Deleting requests cache...")
        delete_cache()

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
        set_torch_seed(torch_random_seed)

    if fewshot_random_seed is not None:
        seed_message.append(f"Setting fewshot manual seed to {fewshot_random_seed}")

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    if tasks is None:
        tasks = []
    if len(tasks) == 0:
        raise ValueError(
            "No tasks specified, or no tasks found. Please verify the task names."
        )

    if gen_kwargs:
        if isinstance(gen_kwargs, str):
            gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            f"generation_kwargs: {gen_kwargs} specified through cli, these settings will update set parameters in yaml tasks. "
            "Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if not gen_kwargs:
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            eval_logger.warning("model_args not specified. Using defaults.")
            model_args = ""

        if isinstance(model_args, dict):
            eval_logger.info(
                f"Initializing {model} model, with arguments: {model_args}"
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )

        else:
            eval_logger.info(
                wrap_text(
                    f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
                )
            )
            lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
                model_args,
                {
                    "batch_size": batch_size,
                    "max_batch_size": max_batch_size,
                    "device": device,
                },
            )
    else:
        if not isinstance(model, lm_eval.api.model.LM):
            raise TypeError(
                f"The value of `model` passed to simple_evaluate() was of type {type(model)}, but is required to be a subclass of lm_eval.api.model.LM . This may be because you are passing an initialized Hugging Face PreTrainedModel without having wrapped it in `lm_eval.models.huggingface.HFLM(pretrained=my_model)` first."
            )
        eval_logger.info("Using pre-initialized model")
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
        metadata = (
            simple_parse_args_string(model_args)
            if isinstance(model_args, str)
            else model_args
            if isinstance(model_args, dict)
            else {}
        ) | (metadata or {})
        task_manager = TaskManager(metadata=metadata)

    # Load tasks - returns {"tasks":.., "groups":..}
    loaded = task_manager.load(tasks)

    # Log selected tasks with hierarchy
    _log_selected_tasks(loaded["tasks"], loaded["groups"], task_manager)

    # Apply config overrides to tasks
    for task_name, task_obj in loaded["tasks"].items():
        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )
            eval_logger.info(
                f"{task_obj.config.task}: Using gen_kwargs: {task_obj.config.generation_kwargs}"
            )

        if predict_only:
            eval_logger.info(
                f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
            )
            # we have to change the class properties post-hoc. This is pretty hacky.
            task_obj.override_metric(metric_name="bypass")

        # override tasks' fewshot values to the provided num_fewshot arg value
        # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
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
        else:
            # if num_fewshot not provided, and the task does not define a default one, default to 0
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                task_obj.set_config(key="num_fewshot", value=0)
        # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
        task_obj.set_fewshot_seed(seed=fewshot_random_seed)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model if isinstance(model, str) else "CUSTOM",
            model_args=model_args or "",
            system_instruction=system_instruction,
            chat_template=lm.chat_template(apply_chat_template)
            if apply_chat_template
            else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    results = evaluate(
        lm=lm,
        task_dict=loaded,
        limit=limit,
        samples=samples,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        bootstrap_iters=bootstrap_iters,
        write_out=write_out,
        log_samples=True if predict_only else log_samples,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        verbosity=verbosity,
        confirm_run_unsafe_code=confirm_run_unsafe_code,
    )
    if verbosity is not None:
        setup_logging(verbosity=verbosity)

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
        }
        # add more detailed model info if available
        if hasattr(lm, "get_model_info"):
            results["config"].update(lm.get_model_info())  # type: ignore
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (
                    list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []  # type: ignore
                ),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": gen_kwargs,
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        results["git_hash"] = get_git_commit_hash()
        results["date"] = start_date
        add_env_info(results)  # additional environment info to results
        add_tokenizer_info(results, lm)  # additional info about tokenizer
        return results
    else:
        return None


@positional_deprecated
def evaluate(
    lm: LM,
    task_dict: TaskDict | _NestedDict,
    limit: int | None = None,
    samples: dict[str, list[int]] | None = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: int | None = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: str | None = None,
    apply_chat_template: bool | str = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    confirm_run_unsafe_code: bool = False,
) -> EvalResults | None:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        lm (LM): Language Model.
        task_dict (TaskDict): Dictionary returned by TaskManager.load() containing
            'tasks', 'groups', and 'group_map' entries.
        limit (int | None): Limit the number of examples per task (only use this
            for testing).
        samples (dict | None): Dictionary indicating which examples should be
            tested in each task, e.g.,
            {"mmlu_astronomy": [0, 3, 6], "mmlu_anatomy": [1, 4, 7, 10]}.
        cache_requests (bool): Speed up evaluation by caching the building of
            dataset requests.
        rewrite_requests_cache (bool): Rewrites all the request cache if set to
            `True`.
        bootstrap_iters (int | None): Number of iterations for bootstrap
            statistics, used when calculating stderr. Set to 0 for skipping all
            stderr calculations.
        write_out (bool): If True, write out an example document and model input
            for checking task integrity.
        log_samples (bool): If True, write out all model outputs and documents
            for per-sample measurement and post-hoc analysis.
        system_instruction (str | None): System instruction to be applied to the
            prompt.
        apply_chat_template (bool | str): Specifies whether to apply a chat
            template to the prompt. If set to True, the default chat template is
            applied. If set to a string, applies the specified chat template by
            name. Defaults to False (no chat template applied).
        fewshot_as_multiturn (bool): Whether to provide the fewshot examples as a
            multiturn conversation or a single user turn.
        verbosity (str): Verbosity level for logging. (no-op, deprecated)
        confirm_run_unsafe_code (bool): Whether to confirm running tasks marked
            as unsafe (e.g code execution tasks).

    Returns:
        dict | None: Dictionary of results, or None if not on rank 0.
    """

    if limit is not None and samples is not None:
        raise ValueError(
            "Either 'limit' or 'samples' must be None, but both are not None."
        )
    if samples is not None:
        eval_logger.info(f"Evaluating examples for tasks {list(samples.keys())}")
    # tracks all Instances/requests a model must generate output on.
    requests = defaultdict(list)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = defaultdict(int)

    # Initialize groups and tasks
    # handle back_comp. Assume if "tasks" not present, then using old nested.
    if "tasks" not in task_dict:
        groups, eval_tasks = _handle_back_comp(cast("_NestedDict", task_dict))
    else:
        task_dict = cast("TaskDict", task_dict)
        groups, eval_tasks = task_dict.get("groups", {}), task_dict.get("tasks", {})

    # Initialize accumulators for per-sample metrics and logged samples
    eval_results_acc: dict[str, ResultAcc] = {
        task_name: {
            "task": task_obj,
            "raw_metrics": defaultdict(list),
            "logged_samples": [],
        }
        for task_name, task_obj in eval_tasks.items()
    }
    if not log_samples and not all(
        "bypass" not in getattr(task_obj, "_metric_fn_list", {})
        for task_obj in eval_tasks.values()
    ):
        raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    # validation checks:
    # 1.are we running code that is marked as unsafe.
    # 2.are we running multimodal task <-> non-multimodal model class, or vice-versa.
    incompatible_tasks = []
    for task_name, task in eval_tasks.items():
        if getattr(task, "UNSAFE_CODE", False) and not confirm_run_unsafe_code:
            raise ValueError(
                f"Attempted to run task: {task_name} which is marked as unsafe. Set confirm_run_unsafe_code=True to run this task."
            )

        if getattr(task, "MULTIMODAL", False) and not getattr(lm, "MULTIMODAL", False):
            incompatible_tasks.append(task_name)
    if len(incompatible_tasks) > 0 and not getattr(lm, "MULTIMODAL", False):
        raise ValueError(
            f"Attempted to run tasks: {incompatible_tasks} which require multimodal input, but the selected model type does not currently implement this. Multimodal support is currently restricted to the ['hf-multimodal', 'vllm-vlm'] model type."
        )
    # end validation check

    # Cache the limit arg.
    limit_arg = limit
    limits = []
    for task_name, task in eval_tasks.items():
        limit = get_sample_size(task, limit_arg)
        limits.append(limit)
        task.build_all_requests(
            limit=limit,
            samples=samples.get(task_name, None) if samples is not None else samples,
            rank=lm.rank,
            world_size=lm.world_size,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template", None)
            if apply_chat_template
            else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "")
            if apply_chat_template
            else "",
        )
        eval_logger.debug(
            f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
        )
        if write_out:
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            import torch

            instances_rnk = torch.tensor(
                len(task._instances) if task._instances else 0, device=lm.device
            )
            gathered_item = lm.all_gather(instances_rnk).cpu().detach().numpy().tolist()
            # "multiple_choice" task types dispatch (several) "loglikelihood" request types
            reqtype = (
                "loglikelihood"
                if task.OUTPUT_TYPE == "multiple_choice"
                else task.OUTPUT_TYPE
            )
            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            # todo: may not account for padding in cases like SquadV2 which has multiple req types
            padding_requests[reqtype] += numpad

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
        for x, req in zip(resps, cloned_reqs, strict=True):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.barrier()

    RANK = lm.rank
    WORLD_SIZE = lm.world_size
    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for (task_name, acc), limit in zip(eval_results_acc.items(), limits, strict=True):
        task = acc["task"]
        task.apply_filters()

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps:
            indices = samples.get(task_name, None) if samples is not None else None
            doc_iterator = task.doc_iterator(
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
                samples=indices,
            )
            for doc_id, doc in doc_iterator:
                doc_id_true = indices[doc_id] if indices else doc_id
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id_true,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "filter": filter_key,
                        "metrics": list(metrics.keys()),
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    acc["logged_samples"].append(example)
                for metric, value in metrics.items():
                    acc["raw_metrics"][(metric, filter_key)].append(value)

    if WORLD_SIZE > 1:
        # Gather all sample metrics across ranks, keyed by task name.
        if log_samples:
            rank_samples = {
                task_name: acc["logged_samples"]
                for task_name, acc in eval_results_acc.items()
            }
            all_samples = lm.gather_object(rank_samples, dst=0)
            if RANK == 0:
                for task_name, acc in eval_results_acc.items():
                    acc["logged_samples"] = list(
                        itertools.chain.from_iterable(
                            rank_data[task_name]
                            for rank_data in all_samples  # type: ignore
                        )
                    )

        rank_metrics = {
            task_name: dict(acc["raw_metrics"])
            for task_name, acc in eval_results_acc.items()
        }
        all_metrics = lm.gather_object(rank_metrics, dst=0)
        if RANK == 0:
            for task_name, acc in eval_results_acc.items():
                for metric_key in acc["raw_metrics"]:
                    acc["raw_metrics"][metric_key] = list(
                        itertools.chain.from_iterable(
                            rank_data[task_name][metric_key]
                            for rank_data in all_metrics  # type: ignore
                        )
                    )

    if RANK == 0:
        res = _process_results(eval_results_acc, groups, bootstrap_iters)

        samples = None
        if log_samples:
            samples = res.samples
            if LMEVAL_HASHMM and hasattr(lm, "MULTIMODAL"):
                samples = hash_dict_images(samples)

        return res._to_eval_results(samples=samples)
    else:
        return None
