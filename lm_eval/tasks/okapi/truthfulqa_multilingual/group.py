#!/usr/bin/env python3
"""
Generate group YAML files for truthfulqa_multilingual tasks, create truthfulqa_multilingual_mc1.yaml and truthfulqa_multilingual_mc2.yaml
with all available language variants.
"""

import glob
import os
import re
from pathlib import Path


def get_language_codes():
    """Extract all language codes from the truthfulqa multilingual files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Get all yaml files in the truthfulqa_multilingual directory
    yaml_files = glob.glob(str(script_dir / "truthfulqa_*_mc*.yaml"))

    # Extract language codes using regex
    language_codes = set()
    for file in yaml_files:
        # Match pattern: truthfulqa_<lang>_mc<number>.yaml
        match = re.match(
            r"truthfulqa_([a-z]{2})_mc([12])\.yaml", os.path.basename(file)
        )
        if match:
            lang_code = match.group(1)
            language_codes.add(lang_code)

    return sorted(list(language_codes))


def create_group_file(mc_version):
    """Create a group YAML file for the specified mc version."""
    language_codes = get_language_codes()

    # Create task list
    tasks = [f"truthfulqa_{lang}_mc{mc_version}" for lang in language_codes]

    # Create the YAML content
    yaml_content = f"""group: truthfulqa_multilingual_mc{mc_version}
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
metadata:
  version: 1.0
"""

    return yaml_content


def main():
    """Main function to generate both mc1 and mc2 group files."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Create mc1 group file
    mc1_content = create_group_file(1)
    mc1_path = script_dir / "truthfulqa_multilingual_mc1.yaml"
    with open(mc1_path, "w") as f:
        f.write(mc1_content)

    # Create mc2 group file
    mc2_content = create_group_file(2)
    mc2_path = script_dir / "truthfulqa_multilingual_mc2.yaml"
    with open(mc2_path, "w") as f:
        f.write(mc2_content)


if __name__ == "__main__":
    main()
