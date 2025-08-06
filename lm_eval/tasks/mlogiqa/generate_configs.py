#!/usr/bin/env python3
"""Generate MLogiQA language-specific YAML configurations."""

import os
from pathlib import Path

import yaml
from datasets import get_dataset_config_names
from dotenv import load_dotenv


def main():
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not found in .env file")

    languages = get_dataset_config_names("swiss-ai/mlogiqa", token=token)
    output_dir = Path(__file__).parent

    for language in sorted(languages):
        config = {
            "task": f"mlogiqa_{language}",
            "include": "_mlogiqa_yaml",
            "dataset_name": language,
        }

        output_file = output_dir / f"mlogiqa_{language}.yaml"
        with open(output_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Created: {output_file}")


if __name__ == "__main__":
    main()
