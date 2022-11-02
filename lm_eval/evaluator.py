import collections
import itertools
import json
import logging
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Optional

import lm_eval.models
import lm_eval.tasks
import lm_eval.api.metric
import lm_eval.api.model
from lm_eval.api.utils import DEFAULT_SEED, set_seed
from lm_eval.api.task import Task


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def cli_evaluate(
    *,
    model_api_name: str,
    model_args: str,
    task_name: str,
    task_args: str,
    template_names: List[str],
    num_fewshot: Optional[int] = 0,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[bool] = False,
    bootstrap_iters: Optional[int] = 100000,
    seed: Optional[int] = DEFAULT_SEED,
    limit: Optional[int] = None,
) -> dict:
    """Evaluate a model from an api on a given task with multiple possible prompt
    formats. This is effectively a wrapper around `evaluate` for command-line
    interface (CLI) like usage; only primitive type arguments.

    Args:
        model_api_name (str):
            Name of the language model api to use. See:
                `lm_eval.models.list_model_apis`
        model_args (str):
            String arguments for the model api. See:
                `lm_eval.api.model.get_model_from_args_string`
        task_name (str):
            The task name of the task to evaluate the model on.
        task_args (str):
            String arguments for the task. See:
                `lm_eval.api.task.get_task_list_from_args_string`
            WARNING: To avoid parse errors, separators must not contain commas.
        template_names (List[str]):
            List of template names for the specified `task_name` to evaluate
            under.
        num_fewshot (int, optional, defaults to 0):
            Number of examples in few-shot context.
        batch_size (int, optional, defaults to None):
            Batch size to use for model evaluation.
        device (str, optional, defaults to None):
            PyTorch device (e.g. "cpu" or "cuda:0") for running models.
        use_cache (bool, optional, defaults to False):
            Whether or not to use a cache for language model results.
        bootstrap_iters (int, optional, defaults to 100000):
            Number of iterations for bootstrap statistics.
        seed (int, optional, defaults to 1234 = `DEFAULT_SEED`):
            Seed for pseudo-random number generation. This controls document
            shuffling, few-shot prompt selection, and framework seeding.
        limit (int, optional, defaults to None):
            Limit the number of examples per task (only use this for testing).

    Returns:
        Dictionary of results.
    """
    tasks = lm_eval.tasks.get_task_list_from_args_string(
        task_name, template_names, task_args
    )
    model = lm_eval.models.get_model_from_args_string(
        model_api_name, model_args, {"batch_size": batch_size, "device": device}
    )

    if use_cache:
        cache_args = model_args.replace("=", "-").replace(",", "_").replace("/", "-")
        # TODO: Make `cache_location` path configurable thru an environment var.
        cache_location = f"lm_cache/{model_api_name}_{cache_args}.db"
        model = lm_eval.api.model.CachingLM(model, cache_location)

    results = evaluate(
        model=model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
        limit=limit,
    )

    # Add info about the model and few shot config.
    results["config"] = {
        "model": model_api_name,
        "model_args": model_args,
        "task_args": task_args,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "use_cache": use_cache,
        "limit": limit,
        "bootstrap_iters": bootstrap_iters,
        "seed": seed,
    }
    return results


def evaluate(
    *,
    model: lm_eval.api.model.LM,
    tasks: List[Task],
    num_fewshot: Optional[int] = 0,
    bootstrap_iters: Optional[int] = 100000,
    seed: Optional[int] = DEFAULT_SEED,
    limit: Optional[int] = None,
) -> dict:
    """Instantiate and evaluate a model on a list of tasks.

    Args:
        model (lm_eval.api.model.LM):
            Language model API instance.
        tasks (List[Task]):
            List of tasks to evaluate `model` on.
        num_fewshot (int, optional, defaults to 0):
            Number of examples in the few-shot context.
        bootstrap_iters (int, optional, defaults to 100000):
            Number of iterations for bootstrap statistics.
        seed (int, optional, defaults to 1234 = `DEFAULT_SEED`):
            Seed for pseudo-random number generation. This controls document
            shuffling, few-shot prompt selection, and framework seeding.
        limit (int, optional, defaults to None):
            Limit the number of examples per task.
            WARNING: This is only for testing purposes.

    Returns:
        Dictionary of results.
    """
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # TODO: Completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces
    task_dict = {}
    for task in tasks:
        if task.has_validation_docs() is False and task.has_test_docs() is False:
            logger.info(
                f"Ignoring Task: {lm_eval.tasks.get_registry_name_from_task(task)} has no validation or test docs"
            )
            continue
        # Create unique keys for each task-template pair.
        task_name = lm_eval.tasks.get_registry_name_from_task(task)
        template_name = task.prompt_template.name if task.prompt_template else None
        key = lm_eval.tasks._get_task_template_key(task_name, template_name)
        task_dict[key] = task

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # TODO: We need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    # Build contexts and collect language model requests.
    for task_template_key, task in task_dict.items():
        task_docs = task.evaluation_docs()

        logger.info(f"\n» Assigning unique IDs to '{task_template_key}' docs")
        task_docs = task_docs.map(
            lambda ex, idx: {**ex, "doc_id": idx}, with_indices=True
        )

        logger.info(f"\n» Filtering invalid docs from '{task_template_key}'")
        task_docs = task_docs.filter(lambda d: not task.invalid_doc_for_prompt(d))
        task_docs = task_docs.shuffle(generator=rng)

        logger.info(f"\n» Constructing '{task_template_key}' contexts and requests")
        pbar_limit = len(task_docs) if not limit else np.minimum(limit, len(task_docs))

        for doc_id, doc in enumerate(
            tqdm(itertools.islice(task_docs, 0, limit), total=pbar_limit)
        ):
            docs[(task_template_key, doc_id)] = doc
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
                # i: Index in requests for a single task instance
                # doc_id: Unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append(
                    (i, task_template_key, doc, doc_id, fewshotex_logging_info)
                )
        # Store the task version.
        versions[task_template_key] = task.VERSION

    # All responses for each (task, doc)
    process_response_queue = collections.defaultdict(list)
    # Execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: Right now, this code runs multiple separate LM requests for
        # multiple Requests differing only in index. We could implement some
        # kind of caching, but that would be more of a band-aid solution. We
        # could also implement some kind of auto-grouping here; they should
        # end up next to each other.
        logger.info(f"\n» Running all `{reqtype}` requests")
        resps = getattr(model, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]
        for resp, (i, task_template_key, doc, doc_id, fewshotex_logging_info) in zip(
            resps, requests_origin[reqtype]
        ):
            process_response_queue[(task_template_key, doc_id)].append(
                (i, resp, fewshotex_logging_info)
            )

    # Unpack results and sort back in order and return control to Task
    vals = collections.defaultdict(list)
    example_logger = logging.getLogger("examples")
    for (task_template_key, doc_id), per_doc_requests in process_response_queue.items():
        per_doc_requests.sort(key=lambda x: x[0])
        per_doc_results = [x[1] for x in per_doc_requests]
        fewshot_logging_info = [x[2] for x in per_doc_requests][0]

        task = task_dict[task_template_key]
        doc = docs[(task_template_key, doc_id)]

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
            vals[(task_template_key, metric)].append(value)

    # Aggregate results
    metric_results = []
    for (task_template_key, metric), items in vals.items():
        task_name, prompt_name = lm_eval.tasks._split_task_template_key(
            task_template_key
        )

        results[task_template_key]["task_name"] = task_name
        results[task_template_key]["prompt_name"] = prompt_name
        task = task_dict[task_template_key]
        results[task_template_key][metric] = task.aggregation()[metric](items)

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
            results[task_template_key][metric + "_stderr"] = stderr(items)
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
    """Returns a markdown table from an evaluation results `dict`.

    Args:
        results (dict):
            A dict of results as found in the `"table_results"` key of the
            dictionary returned by `evaluate`.

    Returns:
        The markdown table of results as a string.
    """
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
