#!/usr/bin/env python3
"""
Generate YAML configs for PolygloToxicityPrompts dataset.

The dataset has 3 splits: ptp-full, ptp-small, and wildchat containing 25K, 5K and 1K prompts per language respectively.

Each config will be placed in the appropriate folder structure:
- full/ (for ptp-* subset with full split)
- small/ (for ptp-* subset with small split)
- wildchat/ (for wildchat-* subset)

Additionally, group configurations are created to run all tasks in each subset.
"""

from pathlib import Path
from typing import Dict, List

LANGUAGE_NAMES = {
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sv": "swedish",
    "zh": "chinese",
}

CONFIG_DEFINITIONS = {
    "full": {
        "folder": "full",
        "prefix": "polyglotoxicityprompts_full",
        "dataset_prefix": "ptp",
        "split": "full",
    },
    "small": {
        "folder": "small",
        "prefix": "polyglotoxicityprompts_small",
        "dataset_prefix": "ptp",
        "split": "small",
    },
    "wildchat": {
        "folder": "wildchat",
        "prefix": "polyglotoxicityprompts_wildchat",
        "dataset_prefix": "wildchat",
        "split": "wildchat",
    },
}


def get_language_name(code: str) -> str:
    """Convert language code to full name."""
    return LANGUAGE_NAMES.get(code, code)


def create_yaml_content(task_name: str, dataset_name: str, test_split: str) -> str:
    """Create YAML content for a specific configuration."""
    return f"""task: {task_name}
include: ../_polyglotoxicityprompts_yaml
dataset_name: {dataset_name}
test_split: {test_split}
"""


def create_group_yaml_content(group_name: str, task_list: List[str]) -> str:
    """Create YAML content for a group configuration."""
    tasks_yaml = "\n".join([f"  - {task}" for task in task_list])
    return f"""group: {group_name}
task:
{tasks_yaml}
aggregate_metric_list:
  - metric: score
    aggregation: mean
    weight_by_size: true
  - metric: perspective_api_toxicity_score
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
"""


def create_directory_structure() -> Path:
    """Create the directory structure for the YAML files."""
    base_dir = Path(__file__).parent

    for config_type in CONFIG_DEFINITIONS:
        folder_path = base_dir / CONFIG_DEFINITIONS[config_type]["folder"]
        folder_path.mkdir(exist_ok=True)

    return base_dir


def generate_configs_for_type(config_type: str, base_dir: Path) -> List[str]:
    """Generate YAML configs for a specific configuration type (full, small, wildchat)."""
    config_def = CONFIG_DEFINITIONS[config_type]
    folder = config_def["folder"]
    prefix = config_def["prefix"]
    dataset_prefix = config_def["dataset_prefix"]
    split = config_def["split"]

    task_names = []

    for lang_code in LANGUAGE_NAMES:
        lang_name = get_language_name(lang_code)
        dataset_name = f"{dataset_prefix}-{lang_code}"
        task_name = f"{prefix}_{lang_name}"

        yaml_content = create_yaml_content(task_name, dataset_name, split)

        file_path = base_dir / folder / f"{task_name}.yaml"
        with open(file_path, "w") as f:
            f.write(yaml_content)

        task_names.append(task_name)

    return task_names


def generate_group_configs(base_dir: Path, task_lists: Dict[str, List[str]]):
    """Generate group YAML configurations."""
    for config_type, task_list in task_lists.items():
        config_def = CONFIG_DEFINITIONS[config_type]
        group_name = f"polyglotoxicityprompts_{config_type}"

        group_content = create_group_yaml_content(group_name, task_list)

        group_path = base_dir / f"{group_name}.yaml"
        with open(group_path, "w") as f:
            f.write(group_content)


def main():
    """Main function to generate all YAML configurations."""

    base_dir = create_directory_structure()
    task_lists = {}

    for config_type in CONFIG_DEFINITIONS:
        task_names = generate_configs_for_type(config_type, base_dir)
        task_lists[config_type] = task_names

    generate_group_configs(base_dir, task_lists)


if __name__ == "__main__":
    main()
