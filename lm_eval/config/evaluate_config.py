import json
import logging
import textwrap
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import yaml

from lm_eval.utils import simple_parse_args_string


if TYPE_CHECKING:
    from lm_eval.tasks import TaskManager

eval_logger = logging.getLogger(__name__)
DICT_KEYS = [
    "wandb_args",
    "wandb_config_args",
    "hf_hub_log_args",
    "metadata",
    "model_args",
    "gen_kwargs",
]


@dataclass
class EvaluatorConfig:
    """Configuration for language model evaluation runs.

    This dataclass contains all parameters for configuring model evaluations via
    `simple_evaluate()` or the CLI. It supports initialization from:
    - CLI arguments (via `from_cli()`)
    - YAML configuration files (via `from_config()`)
    - Direct instantiation with keyword arguments

    The configuration handles argument parsing, validation, and preprocessing
    to ensure properly structured and validated.

    Example:
        # From CLI arguments
        config = EvaluatorConfig.from_cli(args)

        # From YAML file
        config = EvaluatorConfig.from_config("eval_config.yaml")

        # Direct instantiation
        config = EvaluatorConfig(
            model="hf",
            model_args={"pretrained": "gpt2"},
            tasks=["hellaswag", "arc_easy"],
            num_fewshot=5
        )

      See individual field documentation for detailed parameter descriptions.
    """

    # Core evaluation parameters
    config: Optional[str] = field(
        default=None, metadata={"help": "Path to YAML config file"}
    )
    model: str = field(default="hf", metadata={"help": "Name of model e.g. 'hf'"})
    model_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for model initialization"}
    )
    tasks: Union[str, list[str]] = field(
        default_factory=list,
        metadata={"help": "Comma-separated list of task names to evaluate"},
    )

    # Few-shot and batching
    num_fewshot: Optional[int] = field(
        default=None, metadata={"help": "Number of examples in few-shot context"}
    )
    batch_size: int = field(default=1, metadata={"help": "Batch size for evaluation"})
    max_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Maximum batch size for auto batching"}
    )

    # Device
    device: Optional[str] = field(
        default="cuda:0", metadata={"help": "Device to use (e.g. cuda, cuda:0, cpu)"}
    )

    # Data sampling and limiting
    limit: Optional[float] = field(
        default=None, metadata={"help": "Limit number of examples per task"}
    )
    samples: Union[str, dict, None] = field(
        default=None,
        metadata={"help": "dict, JSON string or path to JSON file with doc indices"},
    )

    # Caching
    use_cache: Optional[str] = field(
        default=None,
        metadata={"help": "Path to sqlite db file for caching model outputs"},
    )
    cache_requests: dict = field(
        default_factory=dict,
        metadata={"help": "Cache dataset requests: true/refresh/delete"},
    )

    # Output and logging flags
    check_integrity: bool = field(
        default=False, metadata={"help": "Run test suite for tasks"}
    )
    write_out: bool = field(
        default=False, metadata={"help": "Print prompts for first few documents"}
    )
    log_samples: bool = field(
        default=False, metadata={"help": "Save model outputs and inputs"}
    )
    output_path: Optional[str] = field(
        default=None, metadata={"help": "Dir path where result metrics will be saved"}
    )
    predict_only: bool = field(
        default=False,
        metadata={
            "help": "Only save model outputs, don't evaluate metrics. Use with log_samples."
        },
    )

    # Chat and instruction handling
    system_instruction: Optional[str] = field(
        default=None, metadata={"help": "Custom System instruction to add"}
    )
    apply_chat_template: Union[bool, str] = field(
        default=False,
        metadata={
            "help": "Apply chat template to prompt. Either True, or a string identifying the tokenizer template."
        },
    )
    fewshot_as_multiturn: bool = field(
        default=False,
        metadata={
            "help": "Use fewshot as multi-turn conversation. Requires apply_chat_template=True."
        },
    )

    # Configuration display
    show_config: bool = field(
        default=False, metadata={"help": "Show full config at end of evaluation"}
    )

    # External tasks and generation
    include_path: Optional[str] = field(
        default=None, metadata={"help": "Additional dir path for external tasks"}
    )
    gen_kwargs: Optional[dict] = field(
        default=None, metadata={"help": "Arguments for model generation"}
    )

    # Logging and verbosity
    verbosity: Optional[str] = field(
        default=None, metadata={"help": "Logging verbosity level"}
    )

    # External integrations
    wandb_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for wandb.init"}
    )
    wandb_config_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for wandb.config.update"}
    )
    hf_hub_log_args: dict = field(
        default_factory=dict, metadata={"help": "Arguments for HF Hub logging"}
    )

    # Reproducibility
    seed: list = field(
        default_factory=lambda: [0, 1234, 1234, 1234],
        metadata={"help": "Seeds for random, numpy, torch, fewshot (random)"},
    )

    # Security
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Trust remote code for HF datasets"}
    )
    confirm_run_unsafe_code: bool = field(
        default=False,
        metadata={
            "help": "Confirm understanding of unsafe code risks (for code tasks that executes arbitrary Python)"
        },
    )

    # Internal metadata
    metadata: dict = field(
        default_factory=dict,
        metadata={"help": "Additional metadata for tasks that require it"},
    )

    @classmethod
    def from_cli(cls, namespace: Namespace) -> "EvaluatorConfig":
        """
        Build an EvaluationConfig by merging with simple precedence:
        CLI args > YAML config > built-in defaults
        """
        # Start with built-in defaults
        config = asdict(cls())

        # Load and merge YAML config if provided
        if used_config := hasattr(namespace, "config") and namespace.config:
            config.update(cls.load_yaml_config(namespace.config))

        # Override with CLI args (only truthy values, exclude non-config args)
        excluded_args = {"command", "func"}  # argparse internal args
        cli_args = {
            k: v for k, v in vars(namespace).items() if v and k not in excluded_args
        }
        config.update(cli_args)

        # Parse string arguments that should be dictionaries
        config = cls._parse_dict_args(config)

        # Create an instance and validate
        instance = cls(**config)
        if used_config:
            print(textwrap.dedent(f"""{instance}"""))
        instance.configure()

        return instance

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "EvaluatorConfig":
        """
        Build an EvaluationConfig from a YAML config file.
        Merges with built-in defaults and validates.
        """
        # Load YAML config
        yaml_config = cls.load_yaml_config(config_path)
        # Parse string arguments that should be dictionaries
        yaml_config = cls._parse_dict_args(yaml_config)
        instance = cls(**yaml_config)
        instance.configure()

        return instance

    @staticmethod
    def _parse_dict_args(config: dict[str, Any]) -> dict[str, Any]:
        """Parse string arguments that should be dictionaries."""
        for key in config:
            if key in DICT_KEYS and isinstance(config[key], str):
                config[key] = simple_parse_args_string(config[key])
        return config

    @staticmethod
    def load_yaml_config(config_path: Union[str, Path]) -> dict[str, Any]:
        """Load and validate YAML config file."""
        config_file = (
            Path(config_path) if not isinstance(config_path, Path) else config_path
        )
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            yaml_data = yaml.safe_load(config_file.read_text())
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}") from e
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Could not read config file {config_path}: {e}") from e

        if not isinstance(yaml_data, dict):
            raise ValueError(
                f"YAML root must be a mapping, got {type(yaml_data).__name__}"
            )

        return yaml_data

    def configure(self) -> None:
        """Validate configuration and preprocess fields after creation."""
        self._validate_arguments()
        self._process_arguments()
        self._set_trust_remote_code()

    def _validate_arguments(self) -> None:
        """Validate configuration arguments and cross-field constraints."""
        if self.limit:
            eval_logger.warning(
                "--limit SHOULD ONLY BE USED FOR TESTING. "
                "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )

        # predict_only implies log_samples
        if self.predict_only:
            self.log_samples = True

        # log_samples or predict_only requires output_path
        if (self.log_samples or self.predict_only) and not self.output_path:
            raise ValueError(
                "Specify --output_path if providing --log_samples or --predict_only"
            )

        # fewshot_as_multiturn requires apply_chat_template
        if self.fewshot_as_multiturn and self.apply_chat_template is False:
            raise ValueError(
                "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set."
            )

        # samples and limit are mutually exclusive
        if self.samples and self.limit is not None:
            raise ValueError("If --samples is not None, then --limit must be None.")

        # tasks is required
        if self.tasks is None:
            raise ValueError("Need to specify task to evaluate.")

    def _process_arguments(self) -> None:
        """Process samples argument - load from a file if needed."""
        if self.samples:
            if isinstance(self.samples, dict):
                self.samples = self.samples
            elif isinstance(self.samples, str):
                try:
                    self.samples = json.loads(self.samples)
                except json.JSONDecodeError:
                    if (samples_path := Path(self.samples)).is_file():
                        self.samples = json.loads(samples_path.read_text())

        # Set up metadata by merging model_args and metadata.
        if self.model_args is None:
            self.model_args = {}
        if self.metadata is None:
            self.metadata = {}

        self.metadata = self.model_args | self.metadata

    def process_tasks(self, metadata: Optional[dict] = None) -> "TaskManager":
        """Process and validate tasks, return resolved task names."""
        from lm_eval.tasks import TaskManager

        # if metadata manually passed use that:
        self.metadata = metadata if metadata else self.metadata

        # Create task manager with metadata
        task_manager = TaskManager(
            include_path=self.include_path,
            metadata=self.metadata if self.metadata else {},
        )

        task_names = self.tasks
        # TODO: FIX TASKS VALIDATION!!!
        # task_names = task_manager.match_tasks(self.tasks)

        # # Check for any individual task files in the list
        # for task in [task for task in self.tasks if task not in task_names]:
        #     task_path = Path(task)
        #     if task_path.is_file():
        #         config = utils.load_yaml_config(str(task_path))
        #         task_names.append(config)
        #
        # # Check for missing tasks
        # task_missing = [
        #     task for task in self.tasks if task not in task_names and "*" not in task
        # ]
        #
        # if task_missing:
        #     missing = ", ".join(task_missing)
        #     raise ValueError(f"Tasks not found: {missing}")

        # Update tasks with resolved names
        self.tasks = task_names
        return task_manager

    def _set_trust_remote_code(self) -> None:
        """Apply the trust_remote_code setting if enabled."""
        if self.trust_remote_code:
            # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
            # because it's already been determined based on the prior env var before launching our
            # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

            # Add to model_args for the actual model initialization
            if self.model_args is None:
                self.model_args = {}
            self.model_args["trust_remote_code"] = True
