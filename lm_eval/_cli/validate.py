import argparse
import sys

from lm_eval._cli.base import SubCommand


class ValidateCommand(SubCommand):
    """Command for validating tasks."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the parser
        super().__init__(*args, **kwargs)
        parser = subparsers.add_parser(
            "validate",
            help="Validate task configurations",
            description="Validate task configurations and check for errors.",
            epilog="""
Examples:
  lm-eval validate --tasks hellaswag              # Validate single task
  lm-eval validate --tasks arc_easy,arc_challenge # Validate multiple tasks
  lm-eval validate --tasks mmlu --include_path ./custom_tasks
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add command-specific arguments
        self._add_args(parser)

        # Set the function to execute for this subcommand
        parser.set_defaults(func=self.execute)

    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--tasks",
            "-t",
            required=True,
            type=str,
            metavar="task1,task2",
            help="Comma-separated list of task names to validate",
        )
        parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the validate command."""
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=args.include_path)
        task_list = args.tasks.split(",")

        print(f"Validating tasks: {task_list}")
        # For now, just validate that tasks exist
        task_names = task_manager.match_tasks(task_list)
        task_missing = [task for task in task_list if task not in task_names]

        if task_missing:
            missing = ", ".join(task_missing)
            print(f"Tasks not found: {missing}")
            sys.exit(1)
        else:
            print("All tasks found and valid")
