from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any
from typing_extensions import Self

import yaml

from lm_eval.utils import simple_parse_args_string


if TYPE_CHECKING:
    from argparse import Namespace

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


@dataclass(slots=True)
class EvaluatorConfig:
    """Configuration for language model evaluation runs.

    This dataclass contains all parameters for configuring model evaluations via
    [simple_evaluate][lm_eval.evaluator.simple_evaluate] or the CLI.
    It supports initialization from:

    - CLI arguments (via [from_cli][.from_cli])
    - YAML configuration files (via [from_config][.from_config])
    - Direct instantiation with keyword arguments

    The configuration handles argument parsing, validation, and preprocessing
    to ensure properly structured and validated.

    Example:
        ```python
        # From CLI arguments
        config = EvaluatorConfig.from_cli(args)

        # From YAML file
        config = EvaluatorConfig.from_config("eval_config.yaml")

        # Direct instantiation
        config = EvaluatorConfig(
            model="hf",
            model_args={"pretrained": "gpt2"},
            tasks=["hellaswag", "arc_easy"],
            num_fewshot=5,
        )
        ```
    """

    # Core evaluation parameters

    config: str | None = None
    """Path to a YAML config file. CLI args override values from the file."""

    model: str = "hf"
    """Name of the model backend (e.g. ``"hf"``, ``"vllm"``, ``"openai"``)."""

    model_args: dict = field(default_factory=dict)
    """Arguments for model initialization, passed to the model constructor."""

    tasks: str | list[str] = field(default_factory=list)
    """Task names to evaluate. Accepts a comma-separated string or a list."""

    # Few-shot, repeats, and batching

    num_fewshot: int | None = None
    """Number of examples in few-shot context."""

    repeats: int | None = None
    """Number of repeats for each request (overrides task config)."""

    batch_size: int = 1
    """Batch size for evaluation."""

    max_batch_size: int | None = None
    """Maximum batch size for auto batching."""

    # Device

    device: str | None = "cuda:0"
    """Device to use (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``)."""

    # Data sampling and limiting

    limit: float | None = None
    """Limit number of examples per task. Mutually exclusive with ``samples``."""

    samples: str | dict | None = None
    """Dict, JSON string, or path to a JSON file mapping task names to doc indices."""

    # Caching

    use_cache: str | None = None
    """Path to a SQLite DB file for caching model outputs."""

    cache_requests: dict = field(default_factory=dict)
    """Cache dataset requests. Values: ``true`` / ``"refresh"`` / ``"delete"``."""

    # Output and logging flags

    check_integrity: bool = False
    """Run the test suite for tasks."""

    write_out: bool = False
    """Print prompts for the first few documents."""

    log_samples: bool = False
    """Save model outputs and inputs. Requires ``output_path``."""

    output_path: str | None = None
    """Directory path where result metrics will be saved."""

    predict_only: bool = False
    """Only save model outputs without evaluating metrics. Implies ``log_samples``."""

    # Chat and instruction handling

    system_instruction: str | None = None
    """Custom system instruction prepended to every prompt."""

    apply_chat_template: bool | str = False
    """Apply chat template to the prompt. Either ``True``, or a string naming the tokenizer template."""

    fewshot_as_multiturn: bool | None = None
    """Use fewshot examples as multi-turn conversation. Defaults to ``True`` when ``apply_chat_template`` is set."""

    # Configuration display

    show_config: bool = False
    """Show the full config at the end of evaluation."""

    # External tasks and generation

    include_path: str | None = None
    """Additional directory path for external tasks."""

    gen_kwargs: dict = field(default_factory=dict)
    """Generation arguments passed to the model. Overrides task-level defaults."""

    # Logging and verbosity

    verbosity: str | None = None
    """Logging verbosity level."""

    # External integrations

    wandb_args: dict = field(default_factory=dict)
    """Arguments for ``wandb.init``."""

    wandb_config_args: dict = field(default_factory=dict)
    """Arguments for ``wandb.config.update``."""

    hf_hub_log_args: dict = field(default_factory=dict)
    """Arguments for HF Hub logging."""

    # Reproducibility

    seed: list = field(default_factory=lambda: [0, 1234, 1234, 1234])
    """Seeds as ``[random, numpy, torch, fewshot]``."""

    # Security

    trust_remote_code: bool = False
    """Trust remote code for HF datasets and models."""

    confirm_run_unsafe_code: bool = False
    """Confirm understanding of unsafe code risks (for tasks that execute arbitrary Python)."""

    # Internal metadata

    metadata: dict = field(default_factory=dict)
    """Additional metadata for tasks that require it."""

    @classmethod
    def from_cli(cls, namespace: Namespace) -> EvaluatorConfig:
        """Build an EvaluationConfig by merging with a simple precedence.

        CLI args > YAML config > built-in defaults.
        """
        # Start with built-in defaults
        config = asdict(cls())

        # Load and merge YAML config if provided
        if used_config := getattr(namespace, "config", None):
            config.update(cls.load_yaml_config(used_config))

        # Override with CLI args (skip None = "not provided", exclude non-config args)
        excluded_args = {"command", "func"}  # argparse internal args
        cli_args = {
            k: v
            for k, v in vars(namespace).items()
            if v is not None and k not in excluded_args
        }
        config.update(cli_args)

        # Create an instance and validate
        instance = cls(**config)._parse_dict_args()
        instance._configure()

        if used_config:
            cli_args.pop("config", None)
            eval_logger.info(
                "CLI args %s will override yaml", cli_args
            ) if cli_args else None
            print(textwrap.dedent(f"""{instance}"""))

        return instance

    @classmethod
    def from_config(cls, config_path: str | Path) -> EvaluatorConfig:
        """Build an EvaluationConfig from a YAML config file.

        Merges with built-in defaults and validates.
        """
        # Load YAML config
        yaml_config = cls.load_yaml_config(config_path)
        return cls(**yaml_config)._configure()

    @staticmethod
    def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
        """Load and validate YAML config file."""
        _config_path = Path(config_path)
        if not _config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {_config_path.resolve()}")

        try:
            yaml_data = yaml.safe_load(_config_path.read_text())
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {_config_path}: {e}") from e
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Could not read config file {_config_path}: {e}") from e

        if not isinstance(yaml_data, dict):
            raise TypeError(
                f"YAML root must be a mapping in {_config_path.resolve()}, got {type(yaml_data).__name__}"
            )

        return yaml_data

    def _parse_dict_args(self) -> Self:
        # Parse string arguments that should be dictionaries
        for f in fields(self):
            if f.type is dict and isinstance(getattr(self, f.name), str):
                setattr(self, f.name, simple_parse_args_string(getattr(self, f.name)))
        return self

    def _configure(self) -> Self:
        """Validate configuration and preprocess fields after creation."""
        self._validate_arguments()._process_arguments()._set_trust_remote_code()

        return self

    def _validate_arguments(self) -> Self:
        """Validate configuration arguments and cross-field constraints."""
        # tasks are required
        if self.tasks is None:
            raise ValueError("Need to specify task to evaluate.")

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

        # Handle fewshot_as_multiturn logic:
        # - If None and apply_chat_template is set, default to True
        # - If explicitly True, require apply_chat_template
        # - If explicitly False, keep it False
        if self.fewshot_as_multiturn is None and self.apply_chat_template:
            eval_logger.info("Using default fewshot_as_multiturn=True.")
            self.fewshot_as_multiturn = True
        elif self.fewshot_as_multiturn is True and not self.apply_chat_template:
            raise ValueError(
                "When `fewshot_as_multiturn` is True, `apply_chat_template` must be set."
            )

        # samples and limit are mutually exclusive
        if self.samples and self.limit is not None:
            raise ValueError("If --samples is not None, then --limit must be None.")

        return self

    def _process_arguments(self) -> Self:
        """Process samples argument - load from a file if needed."""
        if self.samples and isinstance(self.samples, str):
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

        return self

    def process_tasks(self, metadata: dict | None = None) -> TaskManager:
        """Process and validate tasks, return resolved task names.

        Handles:
        - Task names (e.g., "hellaswag", "arc_easy")
        - Custom YAML config files (e.g., "/path/to/task.yaml")
        - Glob patterns (e.g., "/path/to/*.yaml")
        - Directories of YAML files
        """
        import itertools

        from lm_eval.tasks import TaskManager
        from lm_eval.tasks._yaml_loader import load_yaml

        # if metadata manually passed use that:
        self.metadata = metadata or self.metadata

        # Create a task manager with metadata
        task_manager = TaskManager(
            include_path=self.include_path,
            metadata=self.metadata or {},
        )

        # Normalize tasks to a list
        # We still allow tasks in the form task1,task2
        task_list = (
            self.tasks.split(",")
            if isinstance(self.tasks, str)
            else [t for task in self.tasks for t in task.split(",")]
        )

        # Handle directory input
        if len(task_list) == 1 and (yaml_path := Path(task_list[0])).is_dir():
            task_names = []
            for yaml_file in yaml_path.glob("*.yaml"):
                config = load_yaml(yaml_file, resolve_func=False)
                task_names.append(config)
            self.tasks = task_names
            return task_manager

        # Normalize paths and deduplicate
        task_list = [
            str(Path(task).absolute()) if task.endswith(".yaml") else task
            for task in task_list
        ]
        match_dict: dict[str, list] = {}

        # Match each task
        for task in task_list:
            if not task.endswith(".yaml"):
                # Standard task name - match via task manager
                matches = task_manager.match_tasks([task])
            else:
                # Custom config file(s) - support glob patterns
                matches = []
                for yaml_file in Path().glob(task):
                    config = load_yaml(yaml_file, resolve_func=False)
                    matches.append(config)
            match_dict[task] = matches

        # Flatten and deduplicate results
        task_names = []
        for task in itertools.chain.from_iterable(match_dict.values()):
            if task not in task_names:
                task_names.append(task)

        # Check for missing tasks
        task_missing = [task for task, matches in match_dict.items() if not matches]
        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(f"Tasks not found: {missing}")

        # Update tasks with resolved names
        self.tasks = task_names
        return task_manager

    def _set_trust_remote_code(self) -> Self:
        """Apply the trust_remote_code setting if enabled."""
        if self.trust_remote_code:
            # Add to model_args for the actual model initialization
            if self.model_args is None:
                self.model_args = {}
            self.model_args["trust_remote_code"] = True

        return self
