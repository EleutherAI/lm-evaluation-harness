import collections
import itertools
import json
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Mapping, Optional, Union

import lm_eval.models
import lm_eval.tasks
import lm_eval.api.metric
import lm_eval.api.model
import lm_eval.api.task
from lm_eval.api.utils import set_seed


logger = logging.getLogger(__name__)


def simple_evaluate(
    *,
    model: Union[str, lm_eval.api.model.LM],
    model_args: Optional[str] = None,
    tasks: List[Union[str, lm_eval.api.task.Task]] = None,
    num_fewshot: Optional[int] = 0,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    no_cache: Optional[bool] = False,
    bootstrap_iters: Optional[int] = 100000,
    limit: Optional[int] = None,
    seed: Optional[int] = 1234,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, lm_eval.api.model.LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name
        task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param seed: int
        Random seed.
    :return
        Dictionary of results
    """
    assert tasks != [], "No tasks specified"
    set_seed(seed)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args,
            {"batch_size": batch_size, "device": device},
        )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if not no_cache:
        cache_args = model_args.replace("=", "-").replace(",", "_").replace("/", "-")
        # TODO: Make this path configurable thru an environment var.
        cache_location = f"lm_cache/{model}_{cache_args}.db"
        lm = lm_eval.api.model.CachingLM(lm, cache_location)

    task_dict = lm_eval.tasks.get_task_dict_promptsource(tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        rng=np.random.default_rng(seed),
    )

    # Add info about the model and few shot config
    results["config"] = {
        "model": model,
        "model_args": model_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
    }
    return results


def evaluate(
    *,
    lm: lm_eval.api.model.LM,
    task_dict: Mapping[str, lm_eval.api.task.PromptSourceTask],
    num_fewshot: Optional[int] = 0,
    bootstrap_iters: Optional[int] = 100000,
    limit: Optional[int] = None,
    rng: Optional[np.random.Generator] = np.random.default_rng(),
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME
        if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param rng: np.random.Generator
        Random number generator for shuffling documents and sampling few-shot examples.
        Default: np.random.default_rng()
    :return
        Dictionary of results
    """
    # TODO: Completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # TODO: We need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    # Get lists of each type of request
    for task_prompt_name, task in task_dict_items:
        task_docs = task.evaluation_docs()
        logger.warning(f"\n» Assigning unique IDs to '{task_prompt_name}' docs")
        task_docs = task_docs.map(
            lambda ex, idx: {**ex, "doc_id": idx}, with_indices=True
        )
        logger.warning(f"» Filtering invalid docs from '{task_prompt_name}'")
        task_docs = task_docs.filter(lambda d: not task.invalid_doc_for_prompt(d))
        task_docs = task_docs.shuffle(generator=rng)
        logger.warning(f"» Constructing '{task_prompt_name}' contexts and requests")
        pbar_limit = len(task_docs) if not limit else np.minimum(limit, len(task_docs))
        for doc_id, doc in enumerate(
            tqdm(itertools.islice(task_docs, 0, limit), total=pbar_limit)
        ):
            docs[(task_prompt_name, doc_id)] = doc
            ctx, fewshotex_logging_info = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                rng=rng,
            )
            fewshotex_logging_info["doc_id"] = doc["doc_id"]
            args = {"num_fewshot": num_fewshot}
            reqs = task.construct_requests(doc, ctx, args)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append(
                    (i, task_prompt_name, doc, doc_id, fewshotex_logging_info)
                )
        # Store the task version.
        versions[task_prompt_name] = task.VERSION

    # All responses for each (task, doc)
    process_response_queue = collections.defaultdict(list)
    # Execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: Right now, this code runs multiple separate LM requests for
        # multiple Requests differing only in index. We could implement some
        # kind of caching, but that would be more of a band-aid solution. We
        # could also implement some kind of auto-grouping here; they should
        # end up next to each other.
        logger.warning(f"\n» Running all `{reqtype}` requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]
        for resp, (i, task_prompt_name, doc, doc_id, fewshotex_logging_info) in zip(
            resps, requests_origin[reqtype]
        ):
            process_response_queue[(task_prompt_name, doc_id)].append(
                (i, resp, fewshotex_logging_info)
            )

    # Unpack results and sort back in order and return control to Task
    vals = collections.defaultdict(list)
    example_logger = logging.getLogger("examples")
    for (task_prompt_name, doc_id), per_doc_requests in process_response_queue.items():
        per_doc_requests.sort(key=lambda x: x[0])
        per_doc_results = [x[1] for x in per_doc_requests]
        fewshot_logging_info = [x[2] for x in per_doc_requests][0]

        task = task_dict[task_prompt_name]
        doc = docs[(task_prompt_name, doc_id)]

        output = task.process_results(doc, per_doc_results)

        if task.save_examples:
            metrics, example = output
            example.update(fewshot_logging_info)
            example.update(task.get_logging_info())
            example_logger.info(json.dumps(example))
        else:
            metrics = output
            example = fewshot_logging_info
            example.update(task.get_logging_info())
            example_logger.info(json.dumps(example))

        for metric, value in metrics.items():
            vals[(task_prompt_name, metric)].append(value)

    # Aggregate results
    metric_results = []
    for (task_prompt_name, metric), items in vals.items():
        task_name, prompt_name = task_prompt_name.split("+", 1)

        results[task_prompt_name]["task_name"] = task_name
        results[task_prompt_name]["prompt_name"] = prompt_name
        task = task_dict[task_prompt_name]
        results[task_prompt_name][metric] = task.aggregation()[metric](items)

        _metric_results = {
            "task_name": task_name,
            "prompt_name": prompt_name,
            metric: task.aggregation()[metric](items),
            **task.get_logging_info(),
        }
        # NOTE: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations.
        # TODO: Find an efficient work around.
        stderr = lm_eval.api.metric.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )
        if stderr is not None:
            results[task_prompt_name][metric + "_stderr"] = stderr(items)
            _metric_results[metric + "_stderr"] = stderr(items)
        metric_results.append(_metric_results)

    return {
        # List of results that tracks the averages per model and prompt.
        "results": metric_results,
        "versions": dict(versions),
        # List of all prompt x doc examples with additional information in it.
        # Original results used for generating the table when running this file.
        "table_results": dict(results),
    }


def make_table(results: dict) -> str:
    """Returns a markdown table from an evaluation results `dict`."""
    from pytablewriter import MarkdownTableWriter

    md_writer = MarkdownTableWriter()
    md_writer.headers = ["Task", "Prompt", "Version", "Metric", "Value", "", "Stderr"]

    values = []
    for k, result_dict in results["table_results"].items():
        version = results["versions"][k]
        for m, v in result_dict.items():
            if m.endswith("_stderr"):
                continue
            if "_name" in m:
                continue
            if m + "_stderr" in result_dict:
                se = result_dict[m + "_stderr"]
                values.append(
                    [
                        result_dict["task_name"],
                        result_dict["prompt_name"],
                        version,
                        m,
                        "%.4f" % v,
                        "±",
                        "%.4f" % se,
                    ]
                )
            else:
                values.append(
                    [
                        result_dict["task_name"],
                        result_dict["prompt_name"],
                        version,
                        m,
                        "%.4f" % v,
                        "",
                        "",
                    ]
                )
            version = ""
    md_writer.value_matrix = values
    return md_writer.dumps()
