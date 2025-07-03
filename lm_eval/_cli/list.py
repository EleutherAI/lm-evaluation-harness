import argparse

from lm_eval._cli.base import SubCommand


class ListCommand(SubCommand):
    """Command for listing available tasks."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the parser
        super().__init__(*args, **kwargs)
        parser = subparsers.add_parser(
            "list",
            help="List available tasks, groups, subtasks, or tags",
            description="List available tasks, groups, subtasks, or tags from the evaluation harness.",
            epilog="""
Examples:
  lm-eval list tasks         # List all available tasks
  lm-eval list groups        # List task groups only
  lm-eval list subtasks      # List subtasks only
  lm-eval list tags          # List available tags
  lm-eval list tasks --include_path /path/to/external/tasks
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args(parser)
        parser.set_defaults(func=self.execute)

    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "what",
            choices=["tasks", "groups", "subtasks", "tags"],
            help="What to list: tasks (all), groups, subtasks, or tags",
        )
        parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks.",
        )

    def execute(self, args: argparse.Namespace) -> None:
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
