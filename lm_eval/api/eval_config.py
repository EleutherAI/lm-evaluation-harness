import argparse
import os
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

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
    Simple config container for language-model evaluation.
    No content validation here—just holds whatever comes from YAML or CLI.
    """

    config: Optional[str]
    model: Optional[str]
    model_args: Optional[dict]
    tasks: Optional[str]
    num_fewshot: Optional[int]
    batch_size: Optional[int]
    max_batch_size: Optional[int]
    device: Optional[str]
    output_path: Optional[str]
    limit: Optional[float]
    samples: Optional[str]
    use_cache: Optional[str]
    cache_requests: Optional[str]
    check_integrity: Optional[bool]
    write_out: Optional[bool]
    log_samples: Optional[bool]
    predict_only: Optional[bool]
    system_instruction: Optional[str]
    apply_chat_template: Optional[Union[bool, str]]
    fewshot_as_multiturn: Optional[bool]
    show_config: Optional[bool]
    include_path: Optional[str]
    gen_kwargs: Optional[dict]
    verbosity: Optional[str]
    wandb_args: Optional[dict]
    wandb_config_args: Optional[dict]
    hf_hub_log_args: Optional[dict]
    seed: Optional[list]
    trust_remote_code: Optional[bool]
    confirm_run_unsafe_code: Optional[bool]
    metadata: Optional[dict]
    request_caching_args: Optional[dict] = None

    @staticmethod
    def parse_namespace(
        namespace: argparse.Namespace,
    ) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
        """
        Convert an argparse Namespace object to a dictionary.

        Handles all argument types including boolean flags, lists, and None values.
        Only includes arguments that were explicitly set (ignores defaults unless used).

        Args:
            namespace: The argparse.Namespace object to convert

        Returns:
            Dictionary containing all the namespace arguments and their values
        """

        config = {key: value for key, value in vars(namespace).items()}
        for key in config:
            if key == "_explicit_args":
                continue
            if key in DICT_KEYS:
                if not isinstance(config[key], dict):
                    config[key] = simple_parse_args_string(config[key])
            # if key == "cache_requests":
            #     config[key] = request_caching_arg_to_dict(config[key])
        if "model_args" not in config:
            config["model_args"] = {}
        if "metadata" not in config:
            config["metadata"] = {}

        non_default_args = []
        if hasattr(namespace, "_explicit_args"):
            non_default_args = namespace._explicit_args

        return config, non_default_args

    @staticmethod
    def non_default_update(console_dict, local_dict, non_default_args):
        """
        Update local_dict by taking non-default values from console_dict.

        Args:
            console_dict (dict): The dictionary that potentially provides updates.
            local_dict (dict): The dictionary to be updated.
            non_default_args (list): The list of args passed by user in console.

        Returns:
            dict: The updated local_dict.
        """
        result_config = {}

        for key in set(console_dict.keys()).union(local_dict.keys()):
            if key in non_default_args:
                result_config[key] = console_dict[key]
            else:
                if key in local_dict:
                    result_config[key] = local_dict[key]
                else:
                    result_config[key] = console_dict[key]

        return result_config

    @classmethod
    def from_cli(cls, namespace: Namespace) -> "EvaluationConfig":
        """
        Build an EvaluationConfig by merging:
          1) YAML config (if --config was passed), then
          2) CLI args (only those the user actually provided)
        CLI defaults that weren’t overridden explicitly will be
        overwritten by YAML values if present.
        """
        # 1. Turn Namespace into a dict + know which were explicitly passed
        args_dict, explicit_args = EvaluationConfig.parse_namespace(namespace)

        # 2. Start from YAML if requested
        config_data: Dict[str, Any] = {}
        if "config" in explicit_args and args_dict.get("config"):
            cfg_path = args_dict["config"]
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(f"Config file not found: {cfg_path}")
            try:
                with open(cfg_path, "r") as f:
                    yaml_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in {cfg_path}: {e}")
            if not isinstance(yaml_data, dict):
                raise ValueError(
                    f"YAML root must be a mapping, got {type(yaml_data).__name__}"
                )
            config_data.update(yaml_data)

        # 3. Override with any CLI args the user explicitly passed
        # for key, val in args_dict.items():
        #     if key == "config":
        #         continue
        #     if key in explicit_args:
        #         config_data[key] = val
        print(f"YAML: {config_data}")
        print(f"CLI: {args_dict}")
        dict_config = EvaluationConfig.non_default_update(
            args_dict, config_data, explicit_args
        )

        # 4. Instantiate the config (no further validation here)
        dict_config.pop("_explicit_args", None)
        return cls(**dict_config)


class TrackExplicitAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Create a set on the namespace to track explicitly set arguments if it doesn't exist
        if not hasattr(namespace, "_explicit_args"):
            setattr(namespace, "_explicit_args", set())
        # Record that this argument was explicitly provided
        namespace._explicit_args.add(self.dest)
        setattr(namespace, self.dest, values)


class TrackExplicitStoreTrue(argparse.Action):
    def __init__(
        self, option_strings, dest, nargs=0, const=True, default=False, **kwargs
    ):
        # Ensure that nargs is 0, as required by store_true actions.
        if nargs != 0:
            raise ValueError("nargs for store_true actions must be 0")
        super().__init__(
            option_strings, dest, nargs=0, const=const, default=default, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        # Initialize or update the set of explicitly provided arguments.
        if not hasattr(namespace, "_explicit_args"):
            setattr(namespace, "_explicit_args", set())
        namespace._explicit_args.add(self.dest)
        setattr(namespace, self.dest, self.const)
