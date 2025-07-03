import argparse
import json
import logging
import os
import textwrap
from functools import partial

from lm_eval._cli import SubCommand
from lm_eval._cli.utils import (
    _int_or_none_list_arg_type,
    request_caching_arg_to_dict,
    try_parse_json,
)


class Run(SubCommand):
    """Command for running language model evaluation."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "run",
            help="Run the evaluation harness on specified tasks",
            description="Evaluate language models on various benchmarks and tasks.",
            usage="lm-eval run --model <model> --tasks <task1,task2,...> [options]",
            epilog=textwrap.dedent("""
                examples:
                  # Basic evaluation with HuggingFace model
                  $ lm-eval run --model hf --model_args pretrained=gpt2 --tasks hellaswag

                  # Evaluate on multiple tasks with few-shot examples
                  $ lm-eval run --model vllm --model_args pretrained=EleutherAI/gpt-j-6B --tasks arc_easy,arc_challenge --num_fewshot 5

                  # Evaluation with custom generation parameters
                  $ lm-eval run --model hf --model_args pretrained=gpt2 --tasks lambada --gen_kwargs "temperature=0.8,top_p=0.95"

                  # Use configuration file
                  $ lm-eval run --config my_config.yaml --tasks mmlu

                For more information, see: https://github.com/EleutherAI/lm-evaluation-harness
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=lambda arg: self._parser.print_help())

    def _add_args(self) -> None:
        self._parser = self._parser
        self._parser.add_argument(
            "--config",
            "-C",
            default=None,
            type=str,
            metavar="DIR/file.yaml",
            help="Path to config with all arguments for `lm-eval`",
        )
        self._parser.add_argument(
            "--model",
            "-m",
            type=str,
            default="hf",
            help="Name of model. Default 'hf'",
        )
        self._parser.add_argument(
            "--tasks",
            "-t",
            default=None,
            type=str,
            metavar="task1,task2",
            help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
        )
        self._parser.add_argument(
            "--model_args",
            "-a",
            default=None,
            type=try_parse_json,
            help="""Comma separated string or JSON formatted arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32` or '{"pretrained":"EleutherAI/pythia-160m","dtype":"float32"}'.""",
        )
        self._parser.add_argument(
            "--num_fewshot",
            "-f",
            type=int,
            default=None,
            metavar="N",
            help="Number of examples in few-shot context",
        )
        self._parser.add_argument(
            "--batch_size",
            "-b",
            type=str,
            default=argparse.SUPPRESS,
            metavar="auto|auto:N|N",
            help="Acceptable values are 'auto', 'auto:N' (recompute batchsize N times with time) or N, where N is an integer. Default 1.",
        )
        self._parser.add_argument(
            "--max_batch_size",
            type=int,
            default=None,
            metavar="N",
            help="Maximal batch size to try with --batch_size auto.",
        )
        self._parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Device to use (e.g. cuda, cuda:0, cpu). Model defaults. Default None.",
        )
        self._parser.add_argument(
            "--output_path",
            "-o",
            default=None,
            type=str,
            metavar="DIR|DIR/file.json",
            help="Path where result metrics will be saved. Can be either a directory or a .json file. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
        )
        self._parser.add_argument(
            "--limit",
            "-L",
            type=float,
            default=None,
            metavar="N|0<N<1",
            help="Limit the number of examples per task. "
            "If <1, limit is a percentage of the total number of examples.",
        )
        self._parser.add_argument(
            "--samples",
            "-E",
            default=None,
            type=try_parse_json,
            metavar="/path/to/json",
            help='JSON string or path to JSON file containing doc indices of selected examples to test. Format: {"task_name":[indices],...}',
        )
        self._parser.add_argument(
            "--use_cache",
            "-c",
            type=str,
            default=None,
            metavar="DIR",
            help="A path to a sqlite db file for caching model responses. `None` if not caching.",
        )
        self._parser.add_argument(
            "--cache_requests",
            type=request_caching_arg_to_dict,
            default=None,
            choices=["true", "refresh", "delete"],
            help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
        )
        self._parser.add_argument(
            "--check_integrity",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Whether to run the relevant part of the test suite for the tasks.",
        )
        self._parser.add_argument(
            "--write_out",
            "-w",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Prints the prompt for the first few documents.",
        )
        self._parser.add_argument(
            "--log_samples",
            "-s",
            action="store_true",
            default=argparse.SUPPRESS,
            help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
        )
        self._parser.add_argument(
            "--system_instruction",
            type=str,
            default=None,
            help="System instruction to be used in the prompt",
        )
        self._parser.add_argument(
            "--apply_chat_template",
            type=str,
            nargs="?",
            const=True,
            default=argparse.SUPPRESS,
            help=(
                "If True, apply chat template to the prompt. "
                "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
                "To apply a specific template from the available list of templates, provide the template name as an argument. "
                "E.g. `--apply_chat_template template_name`"
            ),
        )
        self._parser.add_argument(
            "--fewshot_as_multiturn",
            action="store_true",
            default=argparse.SUPPRESS,
            help="If True, uses the fewshot as a multi-turn conversation",
        )
        self._parser.add_argument(
            "--show_config",
            action="store_true",
            default=argparse.SUPPRESS,
            help="If True, shows the the full config of all tasks at the end of the evaluation.",
        )
        self._parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks to include.",
        )
        self._parser.add_argument(
            "--gen_kwargs",
            type=try_parse_json,
            default=None,
            help=(
                "Either comma delimited string or JSON formatted arguments for model generation on greedy_until tasks,"
                """ e.g. '{"do_sample": True, temperature":0.7,"until":["hello"]}' or temperature=0,top_p=0.1."""
            ),
        )
        self._parser.add_argument(
            "--verbosity",
            "-v",
            type=str.upper,
            default=None,
            metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
            help="(Deprecated) Controls logging verbosity level. Use the `LOGLEVEL` environment variable instead. Set to DEBUG for detailed output when testing or adding new task configurations.",
        )
        self._parser.add_argument(
            "--wandb_args",
            type=str,
            default=argparse.SUPPRESS,
            help="Comma separated string arguments passed to wandb.init, e.g. `project=lm-eval,job_type=eval`",
        )
        self._parser.add_argument(
            "--wandb_config_args",
            type=str,
            default=argparse.SUPPRESS,
            help="Comma separated string arguments passed to wandb.config.update. Use this to trace parameters that aren't already traced by default. eg. `lr=0.01,repeats=3`",
        )
        self._parser.add_argument(
            "--hf_hub_log_args",
            type=str,
            default=argparse.SUPPRESS,
            help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
        )
        self._parser.add_argument(
            "--predict_only",
            "-x",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
        )
        default_seed_string = "0,1234,1234,1234"
        self._parser.add_argument(
            "--seed",
            type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
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
        self._parser.add_argument(
            "--trust_remote_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
        )
        self._parser.add_argument(
            "--confirm_run_unsafe_code",
            action="store_true",
            default=argparse.SUPPRESS,
            help="Confirm that you understand the risks of running unsafe code for tasks that require it",
        )
        self._parser.add_argument(
            "--metadata",
            type=json.loads,
            default=None,
            help="""JSON string metadata to pass to task configs, for example '{"max_seq_lengths":[4096,8192]}'. Will be merged with model_args. Can also be set in task config.""",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Runs the evaluation harness with the provided arguments."""
        from lm_eval.config.evaluate_config import EvaluatorConfig

        # Create and validate config (most validation now happens in EvaluationConfig)
        cfg = EvaluatorConfig.from_cli(args)

        from lm_eval import simple_evaluate, utils
        from lm_eval.loggers import EvaluationTracker, WandbLogger
        from lm_eval.utils import handle_non_serializable, make_table

        # Set up logging
        if cfg.wandb_args:
            wandb_logger = WandbLogger(cfg.wandb_args, cfg.wandb_config_args)

        utils.setup_logging(cfg.verbosity)
        eval_logger = logging.getLogger(__name__)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Set up evaluation tracker
        if cfg.output_path:
            cfg.hf_hub_log_args["output_path"] = cfg.output_path

        if os.environ.get("HF_TOKEN", None):
            cfg.hf_hub_log_args["token"] = os.environ.get("HF_TOKEN")

        evaluation_tracker = EvaluationTracker(**cfg.hf_hub_log_args)

        # Create task manager (metadata already set up in config validation)
        task_manager = cfg.process_tasks()

        # Validation warnings (keep these in CLI as they're logging-specific)
        if "push_samples_to_hub" in cfg.hf_hub_log_args and not cfg.log_samples:
            eval_logger.warning(
                "Pushing samples to the Hub requires --log_samples to be set."
            )

        # Log task selection (tasks already processed in config)
        if cfg.include_path is not None:
            eval_logger.info(f"Including path: {cfg.include_path}")
        eval_logger.info(f"Selected Tasks: {cfg.tasks}")

        # Run evaluation
        results = simple_evaluate(
            model=cfg.model,
            model_args=cfg.model_args,
            tasks=cfg.tasks,
            num_fewshot=cfg.num_fewshot,
            batch_size=cfg.batch_size,
            max_batch_size=cfg.max_batch_size,
            device=cfg.device,
            use_cache=cfg.use_cache,
            cache_requests=cfg.cache_requests.get("cache_requests", False),
            rewrite_requests_cache=cfg.cache_requests.get(
                "rewrite_requests_cache", False
            ),
            delete_requests_cache=cfg.cache_requests.get(
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

        # Process results
        if results is not None:
            if cfg.log_samples:
                samples = results.pop("samples")

            dumped = json.dumps(
                results, indent=2, default=handle_non_serializable, ensure_ascii=False
            )
            if cfg.show_config:
                print(dumped)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

            # W&B logging
            if cfg.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if cfg.log_samples:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to W&B failed: {e}")

            # Save results
            evaluation_tracker.save_results_aggregated(
                results=results, samples=samples if cfg.log_samples else None
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

            # Print results
            print(
                f"{cfg.model} ({cfg.model_args}), gen_kwargs: ({cfg.gen_kwargs}), "
                f"limit: {cfg.limit}, num_fewshot: {cfg.num_fewshot}, "
                f"batch_size: {cfg.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

            if cfg.wandb_args:
                wandb_logger.run.finish()
