import argparse
import json
import numpy as np
import random
import itertools
import collections

from lm_eval import models, tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--model_args', default="")
    parser.add_argument('--tasks', default="all_tasks")
    parser.add_argument('--provide_description', action="store_true")
    parser.add_argument('--num_fewshot', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--limit', default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    lm = models.get_model(args.model).create_from_arg_string(args.model_args)
    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    # TODO: fall back to test docs
    task_dict_items = [(name, task) for name, task in task_dict.items() if task.has_validation_docs()]

    results = {}

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    # if we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger memory,
    # we can always modify this plumbing to support that, but i didn't want to include it just yet because overengineering is bad
    # (or we could make it write the requests to disk and then read them back out again - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable

    docs = {}

    for task_name, task in task_dict_items:
        for doc_id, doc in enumerate(itertools.islice(task.validation_docs(), 0, args.limit)):
            docs[(task_name, doc_id)] = doc

            ctx = task.fewshot_context(
                doc=doc,
                provide_description=args.provide_description,
                num_fewshot=args.num_fewshot,
            )

            reqs = task.construct_requests(ctx)

            for i, req in enumerate(reqs):
                requests[req.type].append(req)
                requests_origin[req.type].append((i, task_name, doc, doc_id))

    process_res_queue = collections.defaultdict(list)

    for reqtype, reqs in requests.items():
        resps = getattr(lm, reqtype)([req.args for req in reqs])

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))
    
    vals = collections.defaultdict(list)

    for (task_name, doc_id), args in process_res_queue.items():
        args.sort(lambda x: x[0])
        args = [x[1] for x in args]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, args)
        for metric in metrics:
            results[(task_name, metric['submetric'])] = {
                "higher_is_better": metric["higher_is_better"],
                "aggregation": metric["aggregation"]
            }
            vals[(task_name, metric['submetric'])].append(metric['value'])
    
    for k in results.keys():
        results[k]['value'] = results[k]['aggregation'](vals[k])

        # can't serialize a function
        del results[k]['aggregation']


    dumped = json.dumps(results, indent=2)
    print(dumped)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
