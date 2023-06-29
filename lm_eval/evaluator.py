import random
import itertools
import json
import collections
import logging
import sys

import torch

import numpy as np

import lm_eval.api
import lm_eval.tasks
import lm_eval.models
import lm_eval.api.metrics
import lm_eval.api.registry

from lm_eval.utils import (
    positional_deprecated,
    run_task_tests,
    make_table,
    create_iterator,
    get_git_commit_hash,
)

from lm_eval.logger import eval_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters=100000,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
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
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

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
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank" + str(lm.rank) + ".db",
        )

    task_dict = lm_eval.tasks.get_task_dict(tasks, num_fewshot=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        output_base_path=output_base_path,
    )

    if lm.rank == 0:
        # add info about the model and few shot config
        results["config"] = {
            "model": model
            if isinstance(model, str)
            else model.model.config._name_or_path,
            "model_args": model_args,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
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
    limit=None,
    bootstrap_iters=100000,
    decontamination_ngrams_path=None,
    write_out=False,
    output_base_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """

    # decontaminate = decontamination_ngrams_path is not None

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    configs = collections.defaultdict(dict)
    samples = collections.defaultdict(list)
    requests = collections.defaultdict(list)

    # docs = {}

    # get lists of each type of request
    for task_name, task in task_dict.items():
        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if limit is not None:
            if task.has_test_docs():
                task_docs = task.test_docs()
            elif task.has_validation_docs():
                task_docs = task.validation_docs()
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        task.build_all_requests(limit=limit, rank=lm.rank, world_size=lm.world_size)

        # aggregate Instances by LM method requested to get output.
        reqtype = (
            "loglikelihood"
            if task.OUTPUT_TYPE == "multiple_choice"
            else task.OUTPUT_TYPE
        )  # TODO: this is hacky, fix in task.py
        requests[reqtype].extend(task.instances)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (numpad > 0):
            for _ in range(numpad):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

    if lm.world_size > 1:
        lm.accelerator.wait_for_everyone()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        task.apply_filters()

    ### Collect values of metrics on all datapoints ###
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        # TODO: make it possible to use a different metric per filter
        # iterate over different filters used
        for key in task.instances[0].filtered_resps.keys():
            doc_iterator = (
                itertools.islice(
                    enumerate(task.test_docs()), lm.rank, limit, lm.world_size
                )
                if task.has_test_docs()
                else itertools.islice(
                    enumerate(task.validation_docs()), lm.rank, limit, lm.world_size
                )
            )

            for doc_id, doc in doc_iterator:
                # subset instances to only this document id ; sort by idx
                requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))
                requests.sort(key=lambda x: x.idx)
                metrics = task.process_results(
                    doc, [req.filtered_resps[key] for req in requests]
                )
                target = task.doc_to_target(doc)
                example = {
                    "doc_id": doc_id,
                    "doc": doc,
                    "target": target,
                    "resps": [req.resps for req in requests],
                    "filtered_resps": [req.filtered_resps[key] for req in requests],
                }
                example.update(metrics)
                samples[task_name].append(example)
                for metric, value in metrics.items():
                    vals[(task_name, key, metric)].append(value)

    if lm.world_size > 1:
        # if multigpu, then gather data across all ranks
        vals_torch = collections.defaultdict(list)
        for (task_name, key, metric), items in vals.items():

            numitem = 0
            if type(items[0]) == tuple:
                numitem = len(items[0])

            # distributed gather requires all ranks to have same dimensions
            # so we pad out with float32 min value
            pad_value = torch.finfo(torch.float32).min
            metrics_tensor = torch.tensor(items, device=lm.device)

            original_dtype = metrics_tensor.dtype  # store original dtype
            torch_device_tensor = lm.accelerator.pad_across_processes(
                metrics_tensor.to(torch.float32), pad_index=pad_value
            )
            gathered_item = lm.accelerator.gather(torch_device_tensor)

            if numitem > 0:
                gathered_filtered = gathered_item[gathered_item[:, 0] != pad_value]
            else:
                gathered_filtered = gathered_item[gathered_item != pad_value]

            gathered_item = (
                gathered_filtered.to(original_dtype).cpu().detach().numpy().tolist()
            )
            # reconvert if we were passed a tuple of values
            if numitem > 0:
                gathered_item = [tuple(g) for g in gathered_item]

            if lm.rank == 0:
                vals_torch[(task_name, key, metric)] = gathered_item

        vals = vals_torch

    if lm.rank == 0:
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for (task_name, key, metric), items in vals.items():
            task = task_dict[task_name]
            results[task_name][metric + "," + key] = task.aggregation()[metric](items)

            # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
            # so we run them less iterations. still looking for a cleaner way to do this

            stderr = lm_eval.api.metrics.stderr_for_metric(
                metric=task.aggregation()[metric],
                bootstrap_iters=min(bootstrap_iters, 1000)
                if metric in ["bleu", "chrf", "ter"]
                else bootstrap_iters,
            )

            if stderr is not None:
                results[task_name][metric + "_stderr" + "," + key] = stderr(items)

        return {
            "results": dict(results),
            "configs": dict(configs),
            "versions": dict(versions),
            "samples": samples,
        }

    else:
        return None
