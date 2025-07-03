import argparse
import json
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Union

from lm_eval.api.eval_config import (
    EvaluationConfig,
    TrackExplicitAction,
    TrackExplicitStoreTrue,
)


def try_parse_json(value: str) -> Union[str, dict, None]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value


def _int_or_none_list_arg_type(
    min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","
):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(
            default_items[num_items:]
        )  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(
                    f"Argument '{action.dest}' doesn't have a type specified."
                )
            else:
                continue


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        "-C",
        default=None,
        type=str,
        metavar="DIR/file.yaml",
        action=TrackExplicitAction,
        help="Path to config with all arguments for `lm-eval`",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="hf",
        action=TrackExplicitAction,
        help="Name of model e.g. `hf`",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        action=TrackExplicitAction,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        action=TrackExplicitAction,
        type=try_parse_json,
        help="""Comma separated string or JSON formatted arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32` or '{"pretrained":"EleutherAI/pythia-160m","dtype":"float32"}'""",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        action=TrackExplicitAction,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        action=TrackExplicitAction,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        action=TrackExplicitAction,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        action=TrackExplicitAction,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        action=TrackExplicitAction,
        metavar="DIR|DIR/file.json",
        help="Path where result metrics will be saved. Can be either a directory or a .json file. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        action=TrackExplicitAction,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--samples",
        "-E",
        default=None,
        type=str,
        action=TrackExplicitAction,
        metavar="/path/to/json",
        help='JSON string or path to JSON file containing doc indices of selected examples to test. Format: {"task_name":[indices],...}',
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        action=TrackExplicitAction,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        action=TrackExplicitAction,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action=TrackExplicitStoreTrue,
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action=TrackExplicitStoreTrue,
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action=TrackExplicitStoreTrue,
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        action=TrackExplicitAction,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=str,
        nargs="?",
        action=TrackExplicitAction,
        const=True,
        default=False,
        help=(
            "If True, apply chat template to the prompt. "
            "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
            "To apply a specific template from the available list of templates, provide the template name as an argument. "
            "E.g. `--apply_chat_template template_name`"
        ),
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action=TrackExplicitStoreTrue,
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action=TrackExplicitStoreTrue,
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        action=TrackExplicitAction,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=try_parse_json,
        default=None,
        action=TrackExplicitAction,
        help=(
            "Either comma delimited string or JSON formatted arguments for model generation on greedy_until tasks,"
            """ e.g. '{"temperature":0.7,"until":["hello"]}' or temperature=0,top_p=0.1."""
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default=None,
        action=TrackExplicitAction,
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="(Deprecated) Controls logging verbosity level. Use the `LOGLEVEL` environment variable instead. Set to DEBUG for detailed output when testing or adding new task configurations.",
    )
    parser.add_argument(
        "--wandb_args",
        type=str,
        default="",
        action=TrackExplicitAction,
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval",
    )
    parser.add_argument(
        "--wandb_config_args",
        type=str,
        default="",
        action=TrackExplicitAction,
        help="Comma separated string arguments passed to wandb.config.update. Use this to trace parameters that aren't already traced by default. eg. `lr=0.01,repeats=3",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        action=TrackExplicitAction,
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action=TrackExplicitStoreTrue,
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        action=TrackExplicitAction,
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action=TrackExplicitStoreTrue,
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action=TrackExplicitStoreTrue,
        help="Confirm that you understand the risks of running unsafe code for tasks that require it",
    )
    parser.add_argument(
        "--metadata",
        type=json.loads,
        default=None,
        action=TrackExplicitAction,
        help="""JSON string metadata to pass to task configs, for example '{"max_seq_lengths":[4096,8192]}'. Will be merged with model_args. Can also be set in task config.""",
    )
    return parser


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        parser = setup_parser()
        args = parse_eval_args(parser)

    cfg = EvaluationConfig.from_cli(args)

    # defer loading `lm_eval` submodules for faster CLI load
    from lm_eval import evaluator, utils
    from lm_eval.evaluator import request_caching_arg_to_dict
    from lm_eval.loggers import EvaluationTracker, WandbLogger
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import (
        handle_non_serializable,
        make_table,
    )

    if args.wandb_args:
        wandb_logger = WandbLogger(cfg.wandb_args, cfg.wandb_config_args)

    utils.setup_logging(cfg.verbosity)
    eval_logger = logging.getLogger(__name__)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # update the evaluation tracker args with the output path and the HF token
    if cfg.output_path:
        cfg.hf_hub_log_args["output_path"] = cfg.output_path

    if os.environ.get("HF_TOKEN", None):
        cfg.hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")

    evaluation_tracker_args = cfg.hf_hub_log_args
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if cfg.predict_only:
        cfg.log_samples = True

    if (cfg.log_samples or cfg.predict_only) and not cfg.output_path:
        raise ValueError(
            "Specify --output_path if providing --log_samples or --predict_only"
        )

    if cfg.fewshot_as_multiturn and cfg.apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if cfg.include_path is not None:
        eval_logger.info(f"Including path: {cfg.include_path}")

    metadata = (cfg.model_args) | (cfg.metadata)
    cfg.metadata = metadata

    # task_manager = TaskManager(include_path=config["include_path"], metadata=metadata)
    task_manager = TaskManager(include_path=cfg.include_path, metadata=metadata)

    if "push_samples_to_hub" in evaluation_tracker_args and not cfg.log_samples:
        eval_logger.warning(
            "Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub."
        )

    if cfg.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if cfg.samples:
        assert cfg.limit is None, "If --samples is not None, then --limit must be None."
        if (samples := Path(cfg.samples)).is_file():
            cfg.samples = json.loads(samples.read_text())
        else:
            cfg.samples = json.loads(cfg.samples)

    if cfg.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif cfg.tasks == "list":
        print(task_manager.list_all_tasks())
        sys.exit()
    elif cfg.tasks == "list_groups":
        print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif cfg.tasks == "list_tags":
        print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif cfg.tasks == "list_subtasks":
        print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(cfg.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(cfg.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                cfg = utils.load_yaml_config(yaml_file)
                task_names.append(cfg)
        else:
            task_list = cfg.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    cfg = utils.load_yaml_config(task)
                    task_names.append(cfg)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )
        cfg.tasks = task_names

    # Respect user's value passed in via CLI, otherwise default to True and add to comma-separated model args
    if cfg.trust_remote_code:
        eval_logger.info(
            "Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`"
        )
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets

        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

        cfg.model_args["trust_remote_code"] = True
    (
        eval_logger.info(f"Selected Tasks: {task_names}")
        if eval_logger.getEffectiveLevel() >= logging.INFO
        else print(f"Selected Tasks: {task_names}")
    )

    request_caching_args = request_caching_arg_to_dict(
        cache_requests=cfg.cache_requests
    )
    cfg.request_caching_args = request_caching_args

    results = evaluator.simple_evaluate(
        model=cfg.model,
        model_args=cfg.model_args,
        tasks=cfg.tasks,
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        use_cache=cfg.use_cache,
        cache_requests=cfg.request_caching_args.get("cache_requests", False),
        rewrite_requests_cache=cfg.request_caching_args.get(
            "rewrite_requests_cache", False
        ),
        delete_requests_cache=cfg.request_caching_args.get(
            "delete_requests_cache", False
        ),
        limit=cfg.limit,
        samples=cfg.samples,
        check_integrity=cfg.check_integrity,
        write_out=cfg.write_out,
        log_samples=cfg.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=cfg.system_instruction,
        apply_chat_template=cfg.apply_chat_template,
        fewshot_as_multiturn=cfg.fewshot_as_multiturn,
        gen_kwargs=cfg.gen_kwargs,
        task_manager=task_manager,
        verbosity=cfg.verbosity,
        predict_only=cfg.predict_only,
        random_seed=cfg.seed[0] if cfg.seed else None,
        numpy_random_seed=cfg.seed[1] if cfg.seed else None,
        torch_random_seed=cfg.seed[2] if cfg.seed else None,
        fewshot_random_seed=cfg.seed[3] if cfg.seed else None,
        confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
        metadata=cfg.metadata,
    )

    if results is not None:
        if cfg.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=handle_non_serializable, ensure_ascii=False
        )
        if cfg.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        # Add W&B logging
        if cfg.wandb_args:
            try:
                wandb_logger.post_init(results)
                wandb_logger.log_eval_result()
                if cfg.log_samples:
                    wandb_logger.log_eval_samples(samples)
            except Exception as e:
                eval_logger.info(f"Logging to Weights and Biases failed due to {e}")

        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples if args.log_samples else None
        )

        if cfg.log_samples:
            for task_name, _ in results["configs"].items():
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        if (
            evaluation_tracker.push_results_to_hub
            or evaluation_tracker.push_samples_to_hub
        ):
            evaluation_tracker.recreate_metadata_card()

        print(
            f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
            f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

        if cfg.wandb_args:
            # Tear down wandb run once all the logging is done.
            wandb_logger.run.finish()


if __name__ == "__main__":
    cli_evaluate()
