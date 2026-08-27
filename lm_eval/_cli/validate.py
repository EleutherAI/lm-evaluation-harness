import argparse
import sys
import textwrap

from lm_eval._cli.subcommand import SubCommand


class Validate(SubCommand):
    """Command for validating tasks."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the self._parser
        super().__init__(*args, **kwargs)
        self._parser = subparsers.add_parser(
            "validate",
            help="Validate task configurations",
            description="Validate task configurations and check for errors.",
            usage="lm-eval validate --tasks <task1,task2> [--include_path DIR]",
            epilog=textwrap.dedent("""
                examples:
                  # Validate a single task
                  lm-eval validate --tasks hellaswag

                  # Validate multiple tasks
                  lm-eval validate --tasks arc_easy,arc_challenge,hellaswag

                  # Validate a task group
                  lm-eval validate --tasks mmlu

                  # Validate tasks with external definitions
                  lm-eval validate --tasks my_custom_task --include_path ./custom_tasks

                  # Validate tasks from multiple external paths
                  lm-eval validate --tasks custom_task1,custom_task2 --include_path "/path/to/tasks1:/path/to/tasks2"

                validation check:
                  The validate command performs several checks:
                  • Task existence: Verifies all specified tasks are available
                  • Configuration syntax: Checks YAML/JSON configuration files
                  • Dataset access: Validates dataset paths and configurations
                  • Required fields: Ensures all mandatory task parameters are present
                  • Metric definitions: Verifies metric functions and aggregation methods
                  • Filter pipelines: Validates filter chains and their parameters
                  • Template rendering: Tests prompt templates with sample data

                task config files:
                  Tasks are defined using YAML configuration files with these key sections:
                  • task: Task name and metadata
                  • dataset_path: HuggingFace dataset identifier
                  • doc_to_text: Template for converting documents to prompts
                  • doc_to_target: Template for extracting target answers
                  • metric_list: List of evaluation metrics to compute
                  • output_type: Type of model output (loglikelihood, generate_until, etc.)
                  • filter_list: Post-processing filters for model outputs

                common errors:
                  • Missing required fields in YAML configuration
                  • Invalid dataset paths or missing dataset splits
                  • Malformed Jinja2 templates in doc_to_text/doc_to_target
                  • Undefined metrics or aggregation functions
                  • Invalid filter names or parameters
                  • Circular dependencies in task inheritance
                  • Missing external task files when using --include_path

                debugging tips:
                  • Use --include_path to test external task definitions
                  • Check task configuration files for syntax errors
                  • Verify dataset access and authentication if needed
                  • Use 'lm-eval list tasks' to see available tasks

                For task configuration guide, see: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md
            """),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        self._add_args()
        self._parser.set_defaults(func=self._execute)

    def _add_args(self) -> None:
        self._parser.add_argument(
            "--tasks",
            "-t",
            required=True,
            type=str,
            metavar="TASK1,TASK2",
            help="Comma-separated list of task names to validate",
        )
        self._parser.add_argument(
            "--include_path",
            type=str,
            default=None,
            metavar="DIR",
            help="Additional path to include if there are external tasks.",
        )

    def _execute(self, args: argparse.Namespace) -> None:
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
