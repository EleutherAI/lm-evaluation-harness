#!/usr/bin/env python3
"""
Generate group YAML files for mgsm tasks, create mgsm_direct.yaml, mgsm_en_cot.yaml, and mgsm_native_cot.yaml with all available language variants and correct filters for each mode.
"""

import glob
import os
import re
from pathlib import Path


def get_language_codes():
    """Extract all language codes from the mgsm files."""
    script_dir = Path(__file__).parent
    yaml_files = glob.glob(str(script_dir / "**/mgsm_*.yaml"), recursive=True)

    language_codes = set()
    for file in yaml_files:
        # Match pattern: mgsm_<mode>_<lang>.yaml
        match = re.match(
            r"mgsm_(direct|en_cot|native_cot)_([a-z]{2})\.yaml", os.path.basename(file)
        )
        if match:
            lang_code = match.group(2)
            language_codes.add(lang_code)

    return sorted(list(language_codes))


def create_group_file(mode):
    """Create a group YAML file for the specified mgsm mode."""
    language_codes = get_language_codes()

    if mode == "direct":
        tasks = [f"mgsm_direct_{lang}" for lang in language_codes]
        group_name = "mgsm_direct"
        filters = ["remove_whitespace", "flexible-extract"]
    elif mode == "en_cot":
        tasks = [f"mgsm_en_cot_{lang}" for lang in language_codes]
        group_name = "mgsm_en_cot"
        filters = ["strict-match", "flexible-extract"]
    elif mode == "native_cot":
        tasks = [f"mgsm_native_cot_{lang}" for lang in language_codes]
        group_name = "mgsm_native_cot"
        filters = ["strict-match", "flexible-extract"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    yaml_content = f"""group: {group_name}
task:
"""

    for task in tasks:
        yaml_content += f"  - {task}\n"

    yaml_content += """
aggregate_metric_list:
  - metric: exact_match
    aggregation: mean
    weight_by_size: true
    filter_list:
"""

    for filter_name in filters:
        yaml_content += f"      - {filter_name}\n"

    yaml_content += """metadata:
  version: 1.0
"""

    return yaml_content


def gen_group_yamls():
    """Generate group YAML files for all mgsm modes."""
    script_dir = Path(__file__).parent

    modes = ["direct", "en_cot", "native_cot"]

    for mode in modes:
        group_content = create_group_file(mode)
        group_filename = f"mgsm_{mode}.yaml"
        group_path = script_dir / group_filename

        with open(group_path, "w") as f:
            f.write(group_content)


if __name__ == "__main__":
    gen_group_yamls()
