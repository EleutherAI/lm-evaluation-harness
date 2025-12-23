# noqa
"""Generate MMMLU YAML configs for every locale and subject."""

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "default"
SUBJECTS_PATH = SCRIPT_DIR / "subjects.json"
LANGUAGES_PATH = SCRIPT_DIR / "languages.json"
CATEGORY_ORDER = ["stem", "other", "social_sciences", "humanities"]

LANGUAGE_METRICS = """aggregate_metric_list:
  - metric: acc
    weight_by_size: True
  - metric: acc_norm
    weight_by_size: True
"""

CATEGORY_METRICS = """aggregate_metric_list:
  - metric: acc
    weight_by_size: True
"""


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def description_for(subject: str, display_name: str) -> str:
    subject_text = " ".join(subject.split("_"))
    return (
        f"The following are multiple choice questions (with answers) about {subject_text}"
        f" ({display_name}).\n\n"
    )


def subject_alias(subject: str, display_name: str) -> str:
    return f"{subject.replace('_', ' ')} ({display_name})"


def quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace("\"", "\\\"")
        .replace("\n", "\\n")
    )
    return f"\"{escaped}\""


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def subject_yaml(base_yaml: str, language: dict, subject: str, category: str) -> str:
    dataset_name = language["dataset_name"]
    display_name = language["display_name"]
    slug = language["slug"]
    lines = [
        f"\"include\": {quote(base_yaml)}",
        f"\"dataset_name\": {quote(dataset_name)}",
        f"process_docs: !function utils.process_{subject}",
        f"\"tag\": {quote(f'mmmlu_{slug}_{category}_tasks')}",
        f"\"task\": {quote(f'mmmlu_{slug}_{subject}')}",
        f"\"task_alias\": {quote(subject_alias(subject, display_name))}",
        f"\"description\": {quote(description_for(subject, display_name))}",
    ]
    return "\n".join(lines)


def category_yaml(language: dict, category: str) -> str:
    slug = language["slug"]
    display_name = language["display_name"]
    label = f"{display_name} {category.replace('_', ' ')}"
    header = [
        f"group: mmmlu_{slug}_{category}",
        f"group_alias: {quote(label)}",
        "task:",
        f"  - mmmlu_{slug}_{category}_tasks",
    ]
    return "\n".join(header + [CATEGORY_METRICS, "metadata:", "  version: 1"])


def language_group_yaml(language: dict, categories: Iterable[str]) -> str:
    slug = language["slug"]
    header = [
        f"group: mmmlu_{slug}",
        f"group_alias: {quote(language['display_name'])}",
        "task:",
    ]
    header.extend(f"  - {category}" for category in categories)
    return "\n".join(header + [LANGUAGE_METRICS, "metadata:", "  version: 1"])


def master_group_yaml(language_groups: Iterable[str]) -> str:
    header = [
        "group: mmmlu",
        "group_alias: \"MMMLU\"",
        "task:",
    ]
    header.extend(f"  - {group}" for group in language_groups)
    return "\n".join(header + [LANGUAGE_METRICS, "metadata:", "  version: 1"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_yaml_path",
        default=DEFAULT_OUTPUT_DIR / "_default_template_yaml",
        type=Path,
        help="Path to the shared template YAML",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory to place generated YAML files",
    )
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Optional subset of language slugs to generate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_yaml = args.base_yaml_path.name
    subjects = load_json(SUBJECTS_PATH)
    languages = load_json(LANGUAGES_PATH)
    if args.languages:
        requested = set(args.languages)
        languages = [lang for lang in languages if lang["slug"] in requested]
    language_groups = []
    for language in languages:
        slug = language["slug"]
        for subject, category in subjects.items():
            file_path = args.output_dir / f"mmmlu_{slug}_{subject}.yaml"
            write_file(file_path, subject_yaml(base_yaml, language, subject, category))
        category_names = []
        for category in CATEGORY_ORDER:
            category_names.append(f"mmmlu_{slug}_{category}")
            category_path = args.output_dir / f"_mmmlu_{slug}_{category}.yaml"
            write_file(category_path, category_yaml(language, category))
        language_group = f"mmmlu_{slug}"
        language_path = args.output_dir / f"_mmmlu_{slug}.yaml"
        write_file(language_path, language_group_yaml(language, category_names))
        language_groups.append(language_group)
    master_path = args.output_dir / "_mmmlu.yaml"
    write_file(master_path, master_group_yaml(language_groups))
