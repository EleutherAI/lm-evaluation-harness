import argparse
import datetime
import json
import logging
import os

from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    return parser.parse_args()


def args_to_name(args):
    """Map `args` to file name. If output_path is set, we use that instead."""
    if args.output_path is not None:
        return args.output_path

    def _fix_model_name(model, model_args):
        if model_args == "":
            return model
        elif "pretrained" in model_args:
            # pretrained=google/t5-base-lm-adapt --> google-t5-base-lm-adapt
            return model_args.split("=")[-1].replace("/", "-")
        else:
            print("WARNING: Unprepared for these model args.")
            return f"{model}_{model_args}"

    fields = [
        _fix_model_name(args.model, args.model_args),
        args.tasks,
        str(args.num_fewshot),
        str(args.seed),
        datetime.datetime.now().isoformat(),
    ]
    fields = [f for f in fields if f is not None]

    if args.limit is not None:
        # Do not use limited files for final analysis.
        return f"limited_{args.limit}_" + "_".join(fields)
    return "_".join(fields)


def main():
    args = parse_args()
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        check_integrity=args.check_integrity,
        seed=args.seed,
    )

    output_path = args_to_name(args)
    os.makedirs("./outputs", exist_ok=True)
    with open(f"./outputs/examples-{output_path}.json", "w") as f:
        json.dump(
            {"examples": results["examples"], "config": results["examples"]},
            f,
        )
    with open(f"./outputs/agg-{output_path}.json", "w") as f:
        json.dump({"results": results["results"], "config": results["examples"]}, f)
    # TODO: Rename codecarbon.csv.
    print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
