#!/usr/bin/env python3
"""
Generate YAML configuration files for the PolyMath dataset.

This script creates:
1. Individual YAML files for each language-difficulty pair (using include: _polymath_yaml)
2. Group files for each language (all 4 difficulties)
3. Group files for each difficulty (all 18 languages)
4. Overall group file called "polymath" that includes all difficulty groups

The individual files use the include mechanism to avoid duplication of common configuration.
"""

from pathlib import Path

import yaml

DIFFICULTIES = ["low", "medium", "high", "top"]
LANGUAGES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "te": "Telugu",
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def create_individual_yaml(language: str, difficulty: str) -> dict:
    """Create YAML configuration for a specific language-difficulty pair."""
    return {
        "task": f"polymath_{language}_{difficulty}",
        "include": "_polymath_yaml",
        "dataset_name": language,
        "test_split": difficulty,
    }


def create_language_group(language: str) -> dict:
    """Create group configuration for a specific language (all difficulties)."""
    language_name = LANGUAGES.get(language, language.upper())
    return {
        "group": f"polymath_{language}",
        "group_alias": f"PolyMath {language_name}",
        "task": [
            f"polymath_{language}_low",
            f"polymath_{language}_medium",
            f"polymath_{language}_high",
            f"polymath_{language}_top",
        ],
        "aggregate_metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "weight_by_size": True}
        ],
        "metadata": {"version": "1.0"},
    }


def create_difficulty_group(difficulty: str) -> dict:
    """Create group configuration for a specific difficulty (all languages)."""
    difficulty_name = difficulty.capitalize()
    return {
        "group": f"polymath_{difficulty}",
        "group_alias": f"PolyMath {difficulty_name}",
        "task": [f"polymath_{lang}_{difficulty}" for lang in LANGUAGES.keys()],
        "aggregate_metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "weight_by_size": True}
        ],
        "metadata": {"version": "1.0"},
    }


def create_main_group() -> dict:
    """Create the main PolyMath group configuration."""
    return {
        "group": "polymath",
        "group_alias": "PolyMath",
        "task": [f"polymath_{lang}" for lang in LANGUAGES],
        "aggregate_metric_list": [
            {"metric": "exact_match", "aggregation": "mean", "weight_by_size": True}
        ],
        "metadata": {
            "version": "1.0",
        },
    }


def write_yaml_file(data: dict, filepath: Path):
    """Write YAML data to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(
            data, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def main():
    """Generate all YAML configuration files."""
    # Create output directory
    main_dir = Path(__file__).parent
    main_dir.mkdir(exist_ok=True)

    # Generate individual language-difficulty YAML files
    tasks_dir = main_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    for language in LANGUAGES:
        for difficulty in DIFFICULTIES:
            config = create_individual_yaml(language, difficulty)
            filename = f"polymath_{language}_{difficulty}.yaml"
            filepath = tasks_dir / filename
            write_yaml_file(config, filepath)

    # Generate difficulty group files
    difficulty_dir = main_dir / "grouped_by_difficulty"
    difficulty_dir.mkdir(exist_ok=True)
    for difficulty in DIFFICULTIES:
        config = create_difficulty_group(difficulty)
        filename = f"polymath_{difficulty}.yaml"
        filepath = difficulty_dir / filename
        write_yaml_file(config, filepath)

    # Generate language group files
    language_dir = main_dir / "grouped_by_language"
    language_dir.mkdir(exist_ok=True)
    for language in LANGUAGES:
        config = create_language_group(language)
        filename = f"polymath_{language}.yaml"
        filepath = language_dir / filename
        write_yaml_file(config, filepath)

    # Generate main group file
    config = create_main_group()
    filename = "polymath.yaml"
    filepath = main_dir / filename
    write_yaml_file(config, filepath)


if __name__ == "__main__":
    main()
