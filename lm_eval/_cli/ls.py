import argparse
import textwrap
from typing import get_args

from lm_eval._cli.subcommand import SubCommand
from lm_eval.api.instance import OutputType


class List(SubCommand):
    """Command for listing available tasks."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the parser
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "ls",
            help="List available tasks, groups, subtasks, or tags",
            description="List available tasks, groups, subtasks, or tags from the evaluation harness.",
            epilog=textwrap.dedent("""
                examples:
                  # List all available tasks (includes groups, subtasks, and tags)
                  $ lm-eval ls tasks

                  # List only task groups (like 'mmlu', 'glue', 'superglue')
                  $ lm-eval ls groups

                  # List only individual subtasks (like 'mmlu_abstract_algebra')
                  $ lm-eval ls subtasks

                  # Show generation tasks with their catalog descriptions
                  $ lm-eval ls subtasks --output-type generate_until --show-description

                  # Include external task definitions
                  $ lm-eval ls tasks --include_path /path/to/external/tasks

                  # List tasks from multiple external paths
                  $ lm-eval ls tasks --include_path "/path/to/tasks1:/path/to/tasks2"

                organization:
                  • Groups: Collections of tasks with aggregated metric across subtasks (e.g., 'mmlu')
                  • Subtasks: Individual evaluation tasks (e.g., 'mmlu_anatomy', 'hellaswag')
                  • Tags: Similar to groups but no aggregate metric (e.g., 'reasoning', 'knowledge', 'language')
                  • External Tasks: Custom tasks defined in external directories

                evaluation usage:
                  After listing tasks, use them with the run command!

                For more information tasks configs are defined in https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        self._parser.add_argument(
            "what",
            choices=["tasks", "groups", "subtasks", "tags"],
            nargs="?",
            help="What to list: tasks (all), groups, subtasks, or tags",
        )
        self._parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks.",
        )
        self._parser.add_argument(
            "--output-type",
            choices=get_args(OutputType),
            help="List only individual tasks with this output type (tasks or subtasks).",
        )
        self._parser.add_argument(
            "--show-description",
            action="store_true",
            help="Include the optional metadata.description for each individual task.",
        )

    def _execute(self, args: argparse.Namespace) -> None:
        """Execute the list command."""
        from lm_eval.tasks import TaskManager

        if (args.output_type or args.show_description) and args.what not in (
            "tasks",
            "subtasks",
        ):
            self._parser.error(
                "--output-type and --show-description require 'tasks' or 'subtasks'"
            )
        if args.what is None:
            self._parser.print_help()
            return

        task_manager = TaskManager(include_path=args.include_path)
        print(
            task_manager.list_all_tasks(
                list_groups=args.what in ("tasks", "groups"),
                list_tags=args.what in ("tasks", "tags"),
                list_subtasks=args.what in ("tasks", "subtasks"),
                output_type=args.output_type,
                show_description=args.show_description,
            )
        )
