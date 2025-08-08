#!/usr/bin/env python3
"""
Generate group YAML files for hellaswag_multilingual tasks, create hellaswag_multilingual.yaml with all available language variants.
"""

import glob
import os
import re
from pathlib import Path


def get_language_codes():
    """Extract all language codes from the hellaswag multilingual files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Get all yaml files in the hellaswag_multilingual directory
    yaml_files = glob.glob(str(script_dir / "hellaswag_*.yaml"))

    # Extract language codes using regex
    language_codes = set()
    for file in yaml_files:
        # Match pattern: hellaswag_<lang>.yaml
        match = re.match(r"hellaswag_([a-z]{2})\.yaml", os.path.basename(file))
        if match:
            lang_code = match.group(1)
            language_codes.add(lang_code)

    return sorted(list(language_codes))


def create_group_file():
    """Create a group YAML file for hellaswag_multilingual."""
    language_codes = get_language_codes()

    # Create task list
    tasks = [f"hellaswag_{lang}" for lang in language_codes]

    # Create the YAML content
    yaml_content = """group: hellaswag_multilingual
task:
"""

    # Add each task with proper indentation
    for task in tasks:
        yaml_content += f"  - {task}\n"

    # Add aggregation metrics
    yaml_content += """
aggregate_metric_list:
  - metric: acc
    aggregation: mean
    weight_by_size: true
  - metric: acc_norm
    aggregation: mean
    weight_by_size: true
metadata:
  version: 1.0
"""

    return yaml_content


def main():
    """Main function to generate the hellaswag_multilingual group file."""
    script_dir = Path(__file__).parent

    group_content = create_group_file()
    group_path = script_dir / "hellaswag_multilingual.yaml"
    with open(group_path, "w") as f:
        f.write(group_content)


if __name__ == "__main__":
    main()
