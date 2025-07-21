#!/usr/bin/env python3
"""
Script to generate multijail config files for multiple languages.
Creates individual language configs and a group config that includes all languages.
"""

import os
from pathlib import Path

# Language configurations
LANGUAGES = {
    "zh": "Chinese",
    "it": "Italian",
    "vi": "Vietnamese",
    "ar": "Arabic",
    "ko": "Korean",
    "th": "Thai",
    "bn": "Bengali",
    "sw": "Swahili",
    "jv": "Javanese",
}


def create_language_config(lang_code, lang_name):
    """Create a language-specific config file."""
    config_content = f"""task: multijail_{lang_code}
include: _multijail_yaml
doc_to_text: "{{{{{lang_code}}}}}"
"""
    return config_content


def create_group_config():
    """Create a group config that includes all language tasks."""
    task_list = [f"multijail_{lang_code}" for lang_code in LANGUAGES.keys()]
    task_list.append("multijail_en")  # Include the existing English config

    config_content = f"""group: multijail
task:
{chr(10).join(f"  - {task}" for task in task_list)}
aggregate_metric_list:
  - metric: score
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
"""
    return config_content


def main():
    """Generate all config files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    print("Generating multijail config files...")

    # Create individual language configs
    for lang_code, lang_name in LANGUAGES.items():
        config_content = create_language_config(lang_code, lang_name)
        config_file = script_dir / f"multijail_{lang_code}.yaml"

        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)

        print(f"Created: {config_file}")

    # Create group config
    group_config_content = create_group_config()
    group_config_file = script_dir / "multijail.yaml"

    with open(group_config_file, "w", encoding="utf-8") as f:
        f.write(group_config_content)

    print(f"Created: {group_config_file}")
    print(f"\nGenerated {len(LANGUAGES) + 1} config files:")
    print("- Individual language configs:")
    for lang_code, lang_name in LANGUAGES.items():
        print(f"  - multijail_{lang_code}.yaml ({lang_name})")
    print("- Group config: multijail.yaml")


if __name__ == "__main__":
    main()
