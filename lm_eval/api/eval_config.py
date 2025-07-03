import json
import logging
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from lm_eval.utils import simple_parse_args_string


DICT_KEYS = [
    "wandb_args",
    "wandb_config_args",
    "hf_hub_log_args",
    "metadata",
    "model_args",
]


@dataclass
class EvaluationConfig:
    """
    Simple config container for holding params.
    """

    config: Optional[str] = None
    model: Optional[str] = None
    model_args: Optional[dict] = None
    tasks: Optional[str] = None
    num_fewshot: Optional[int] = None
    batch_size: Optional[int] = None
    max_batch_size: Optional[int] = None
    device: Optional[str] = None
    output_path: Optional[str] = None
    limit: Optional[float] = None
    samples: Optional[str] = None
    use_cache: Optional[str] = None
    cache_requests: Optional[str] = None
    check_integrity: Optional[bool] = None
    write_out: Optional[bool] = None
    log_samples: Optional[bool] = None
    predict_only: Optional[bool] = None
    system_instruction: Optional[str] = None
    apply_chat_template: Optional[Union[bool, str]] = None
    fewshot_as_multiturn: Optional[bool] = None
    show_config: Optional[bool] = None
    include_path: Optional[str] = None
    gen_kwargs: Optional[dict] = None
    verbosity: Optional[str] = None
    wandb_args: Optional[dict] = None
    wandb_config_args: Optional[dict] = None
    hf_hub_log_args: Optional[dict] = None
    seed: Optional[list] = None
    trust_remote_code: Optional[bool] = None
    confirm_run_unsafe_code: Optional[bool] = None
    metadata: Optional[dict] = None
    request_caching_args: Optional[dict] = None

    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default values for all configuration options."""
        return {
            "model": "hf",
            "model_args": {},
            "batch_size": 1,
            "check_integrity": False,
            "write_out": False,
            "log_samples": False,
            "predict_only": False,
            "fewshot_as_multiturn": False,
            "show_config": False,
            "trust_remote_code": False,
            "confirm_run_unsafe_code": False,
            "metadata": {},
            "wandb_args": {},
            "wandb_config_args": {},
            "hf_hub_log_args": {},
            "seed": [0, 1234, 1234, 1234],
        }

    @staticmethod
    def _parse_dict_args(config: Dict[str, Any]) -> Dict[str, Any]:
        """Parse string arguments that should be dictionaries."""
        for key in config:
            if key in DICT_KEYS and isinstance(config[key], str):
                config[key] = simple_parse_args_string(config[key])
        return config

    @classmethod
    def from_cli(cls, namespace: Namespace) -> "EvaluationConfig":
        """
        Build an EvaluationConfig by merging with simple precedence:
        CLI args > YAML config > built-in defaults
        """
        # Start with built-in defaults
        config = cls._get_defaults()

        # Load and merge YAML config if provided
        if hasattr(namespace, "config") and namespace.config:
            config.update(cls._load_yaml_config(namespace.config))

        # Override with CLI args (only non-None values, exclude non-config args)
        excluded_args = {"config", "command", "func"}  # argparse internal args
        cli_args = {
            k: v
            for k, v in vars(namespace).items()
            if v is not None and k not in excluded_args
        }
        config.update(cli_args)

        # Parse string arguments that should be dictionaries
        config = cls._parse_dict_args(config)

        # Create instance and validate
        instance = cls(**config)
        instance.validate_and_preprocess()

        return instance

    @staticmethod
    def _load_yaml_config(config_path: str) -> Dict[str, Any]:
        """Load and validate YAML config file."""
        config_file = Path(config_path)
        if not config_file.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            yaml_data = yaml.safe_load(config_file.read_text())
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
        except (OSError, UnicodeDecodeError) as e:
            raise ValueError(f"Could not read config file {config_path}: {e}")

        if not isinstance(yaml_data, dict):
            raise ValueError(
                f"YAML root must be a mapping, got {type(yaml_data).__name__}"
            )

        return yaml_data

    def validate_and_preprocess(self) -> None:
        """Validate configuration and preprocess fields after creation."""
        self._validate_arguments()
        self._process_samples()
        self._setup_metadata()
        self._apply_trust_remote_code()
        self._process_tasks()

    def _validate_arguments(self) -> None:
        """Validate configuration arguments and cross-field constraints."""
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

    def _process_samples(self) -> None:
        """Process samples argument - load from file if needed."""
        if self.samples:
            if (samples_path := Path(self.samples)).is_file():
                self.samples = json.loads(samples_path.read_text())
            else:
                self.samples = json.loads(self.samples)

    def _process_tasks(self, metadata: Union[dict, str]) -> List[str]:
        """Process and validate tasks, return resolved task names."""
        from lm_eval import utils
        from lm_eval.tasks import TaskManager

        # Create task manager with metadata
        task_manager = TaskManager(
            include_path=self.include_path, metadata=self.metadata
        )

        # self.tasks is a comma-separated string of task names
        task_list = self.tasks.split(",")
        task_names = task_manager.match_tasks(task_list)

        # Check for any individual task files in the list
        for task in [task for task in task_list if task not in task_names]:
            task_path = Path(task)
            if task_path.is_file():
                config = utils.load_yaml_config(str(task_path))
                task_names.append(config)

        # Check for missing tasks
        task_missing = [
            task for task in task_list if task not in task_names and "*" not in task
        ]

        if task_missing:
            missing = ", ".join(task_missing)
            raise ValueError(f"Tasks not found: {missing}")

        # Update tasks with resolved names
        self.tasks = task_names
        return task_names

    def _setup_metadata(self) -> None:
        """Set up metadata by merging model_args and metadata."""
        if self.model_args is None:
            self.model_args = {}
        if self.metadata is None:
            self.metadata = {}

        # Merge model_args and metadata
        merged_metadata = self.model_args | self.metadata
        self.metadata = merged_metadata

    def _apply_trust_remote_code(self) -> None:
        """Apply trust_remote_code setting if enabled."""
        if self.trust_remote_code:
            eval_logger = logging.getLogger(__name__)
            eval_logger.info("Setting HF_DATASETS_TRUST_REMOTE_CODE=true")

            # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
            # because it's already been determined based on the prior env var before launching our
            # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
            import datasets

            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

            # Add to model_args for the actual model initialization
            if self.model_args is None:
                self.model_args = {}
            self.model_args["trust_remote_code"] = True
