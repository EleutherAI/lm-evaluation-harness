import collections
import itertools
import numpy as np
import random
import lm_eval.api.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.api
from lm_eval.utils import positional_deprecated, run_task_tests, make_table


@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    check_integrity=False,
    decontamination_ngrams_path=None,
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
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    task_dict = lm_eval.tasks.get_task_dict(tasks, num_fewshot=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        num_fewshot=num_fewshot,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
    )

    # add info about the model and few shot config
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


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    decontamination_ngrams_path=None,
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
    :return
        Dictionary of results
    """

    decontaminate = decontamination_ngrams_path is not None

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    docs = {}

    # get lists of each type of request
    for task_name, task in task_dict.items():
        versions[task_name] = task.VERSION
    
        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        # task_docs = list(task_doc_func())
        # rnd = random.Random()
        # rnd.seed(42)
        # rnd.shuffle(task_docs)

        # for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):
        task.build_all_requests(limit=limit)
        # aggregate Instances by LM method requested to get output.
        requests[task.OUTPUT_TYPE].extend(task.instances) 
    
    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        print("Running", reqtype, "requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)
        
        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        task.apply_filters()


    ### Collect values of metrics on all datapoints ###
    # TODO: make metric configurable, add metric registry 
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        # calculate values for each filter setup (TODO: make getting list of keys cleaner)
        # TODO: make it possible to use a different metric per key
        for key in task.instances[0].filtered_resps.keys():
            for doc_id, doc in enumerate(itertools.islice(task.test_docs(), 0, limit) if task.has_test_docs() else task.validation_docs()):
                # subset instances to only this document id ; sort by idx
                requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))
                requests.sort(key=lambda x: x.id_)
                metrics = task.process_results(doc, [req.filtered_resps[key] for req in requests])
                for metric, value in metrics.items():
                    vals[(task_name, key, metric)].append(value)
    


    ### Aggregate results over all datapoints ###
    # aggregate results ; run bootstrap CIs
    for (task_name, key, metric), items in vals.items():
        task = task_dict[task_name]
        results[task_name][metric + " - filter=" + key] = task.aggregation()[metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.api.metrics.stderr_for_metric(
            metric=task.aggregation()[metric],
            bootstrap_iters=min(bootstrap_iters, 1000)
            if metric in ["bleu", "chrf", "ter"]
            else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + " - filter=" + key + "_stderr"] = stderr(items)

    return {"results": dict(results), "versions": dict(versions)}
