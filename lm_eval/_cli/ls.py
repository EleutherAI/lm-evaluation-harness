import argparse
import textwrap

from lm_eval._cli.subcommand import SubCommand


class List(SubCommand):
    """Command for listing available tasks."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the parser
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "ls",
            help="List available tasks, groups, subtasks, or tags",
            description="List available tasks, groups, subtasks, or tags from the evaluation harness.",
            usage="lm-eval list [tasks|groups|subtasks|tags] [--include_path DIR]",
            epilog=textwrap.dedent("""
                examples:
                  # List all available tasks (includes groups, subtasks, and tags)
                  $ lm-eval ls tasks

                  # List only task groups (like 'mmlu', 'glue', 'superglue')
                  $ lm-eval ls groups

                  # List only individual subtasks (like 'mmlu_abstract_algebra')
                  $ lm-eval ls subtasks

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

    def _execute(self, args: argparse.Namespace) -> None:
        """Execute the list command."""
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=args.include_path)

        if args.what == "tasks":
            print(task_manager.list_all_tasks())
        elif args.what == "groups":
            print(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        elif args.what == "subtasks":
            print(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        elif args.what == "tags":
            print(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        elif args.what is None:
            self._parser.print_help()
