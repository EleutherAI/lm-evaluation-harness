import argparse
import datetime
import json
import logging
import os

import lm_eval.evaluator as evaluator
from lm_eval.api import utils


logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_api_name",
        required=True,
        help="Name of the model API to use. See `lm_eval.list_model_apis()` for available APIs",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="Model constructor args that you'd pass into a model of type "
        "`--model_api_name`. These must be comma-separated keyword args, e.g. "
        "`key1=value1,key2=value2`, with no spaces",
    )
    parser.add_argument(
        "--task_name",
        required=True,
        help="Name of the task to use as found "
        "in the lm_eval registry. See: `lm_eval.list_tasks()`",
    )
    parser.add_argument(
        "--task_args",
        default="",
        help="""Optional task constructor args that you'd pass into a task class of kind "
        `--task_name`. These must be comma-separated keyword args, e.g.
        `key1=value1,key2=value2`, with no spaces.
        WARNING: To avoid parsing errors, ensure your strings are quoted. For example,
            `example_separator='\\n+++\\n'`
        WARNING: Values must NOT contain commas.""",
    )
    parser.add_argument(
        "--template_names",
        default="all_templates",
        help="""Comma-separated list of template names for the specified
        task. Example:

        `> python main.py ... --task_name rte --template_names imply,mean`

        - Default: `all_templates`
        - General Selectors:
            - `"all_templates"`: Selects all templates for the task
            - `"original_templates"`: Selects only templates that are designed to match the original task
        """,
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to place your model onto, e.g. cuda:0. For large "
        "models available through the HuggingFace Hub you should use `accelerate` "
        "by passing `use_accelerate=True` to `--model_args`",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate on; ONLY USE THIS FOR DEBUGGING PURPOSES",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="""Use output_path as `output_filename`. For example:

    `> python main.py ... --output_path blop`
    # saves files into `outputs/blop.json`

    Warning: You currently cannot change/add folder structure.
    """,
    )
    parser.add_argument(
        "--template_idx",
        type=int,
        default=None,
        help="Choose template by index from available templates",
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=100000,
        help="Iters for stderr computation",
    )
    parser.add_argument(
        "--no_tracking", action="store_true", help="Skip carbon emission tracking"
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to cache your model's predictions or not",
    )
    return parser.parse_args()


def args_to_name(args, separator):
    """Map `args` to file name. If output_path is set, we use that instead."""
    if args.output_path is not None:
        return args.output_path

    def _fix_model_name(model, model_args):
        if model_args == "":
            return model
        elif "pretrained" not in model_args:
            logger.warning("WARNING: Unprepared for these model args.")
            return f"{model}={model_args}"

        for arg in model_args.split(","):
            # Example:
            #   pretrained=google/t5-base-lm-adapt --> google-t5-base-lm-adapt
            if "pretrained" in arg:
                return arg.split("=")[-1].replace("/", "-")

    fields = {
        "model": _fix_model_name(args.model_api_name, args.model_args),
        "task": args.task_name,
        "templates": args.template_names,
        "fewshot": str(args.num_fewshot),
        "batchsize": str(args.batch_size),
        "seed": str(args.seed),
        "timestamp": datetime.datetime.now().isoformat("T", "seconds"),
    }
    fields = [f"{k}={v}" for k, v in fields.items() if v is not None]
    # Some prompts also have "/" in them!
    filename = f"{separator}".join(fields).replace("/", "-")
    if args.limit is not None:
        # Do not use limited files for final analysis.
        return f"limited={args.limit}{separator}" + filename

    return filename


def setup_example_logger(output_path, separator):
    """Sets up a logger that will save each example and prediction."""
    example_logger = logging.getLogger("examples")
    filename = f"./outputs/examples{separator}{output_path}.jsonl"
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(filename)
    handler.setFormatter(formatter)
    example_logger.addHandler(handler)
    example_logger.setLevel(logging.INFO)


def main():
    os.makedirs("./outputs", exist_ok=True)
    args = parse_args()

    if args.limit:
        logger.warning(
            "\n» WARNING: `--limit` SHOULD ONLY BE USED FOR TESTING. REAL METRICS "
            "SHOULD NOT BE COMPUTED USING LIMIT."
        )

    print()  # Ensure a newline after `main` command for readability.

    path_separator = "."
    output_path = args_to_name(args, separator=path_separator)
    setup_example_logger(output_path, path_separator)

    template_names = utils.cli_template_names(
        args.task_name, args.template_names, args.template_idx
    )
    evaluate_args = dict(
        model_api_name=args.model_api_name,
        model_args=args.model_args,
        task_name=args.task_name,
        task_args=args.task_args,
        template_names=template_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        use_cache=args.use_cache,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
        limit=args.limit,
    )
    if args.no_tracking:
        results = evaluator.cli_evaluate(**evaluate_args)
    else:
        from codecarbon import OfflineEmissionsTracker

        with OfflineEmissionsTracker(country_iso_code="FRA", log_level="error"):
            print()  # Add newline between emissions tracker and evaluation logging.
            results = evaluator.cli_evaluate(**evaluate_args)

    with open(f"./outputs/agg{path_separator}{output_path}.json", "w") as f:
        json.dump({"results": results["results"], "config": results["config"]}, f)

    from scripts.agg2slim import agg2slim

    with open(f"./outputs/slim{path_separator}{output_path}.json", "w") as f:
        json.dump(agg2slim(results), f, indent=2)

    print(f"\n{evaluator.make_table(results)}")

    if not args.no_tracking:
        emissions_output_path = f"./outputs/emissions{path_separator}{output_path}.csv"
        os.rename("emissions.csv", emissions_output_path)


if __name__ == "__main__":
    main()
