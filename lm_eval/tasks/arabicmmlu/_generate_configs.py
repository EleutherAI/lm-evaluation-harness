"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")


SUBJECTS = {
    "Driving Test": "other",
    "High Geography": "social_science",
    "High History": "humanities",
    "Islamic Studies": "humanities",
    "Univ Accounting": "social_science",
    "Primary General Knowledge": "other",
    "Univ Political Science": "social_science",
    "Primary Math": "stem",
    "Middle General Knowledge": "other",
    "High Biology": "stem",
    "Primary Natural Science": "stem",
    "High Economics": "social_science",
    "Middle Natural Science": "stem",
    "Middle Geography": "social_science",
    "Primary Social Science": "social_science",
    "Middle Computer Science": "stem",
    "Middle Islamic Studies": "humanities",
    "Primary Computer Science": "stem",
    "High Physics": "stem",
    "Middle Social Science": "social_science",
    "Middle Civics": "social_science",
    "High Computer Science": "stem",
    "General Knowledge": "other",
    "High Civics": "social_science",
    "Prof Law": "humanities",
    "High Islamic Studies": "humanities",
    "Primary Arabic Language": "language",
    "High Arabic Language": "language",
    "Arabic Language (Grammar)": "language",
    "Primary History": "humanities",
    "Middle History": "humanities",
    "Univ Economics": "social_science",
    "Arabic Language (General)": "language",
    "Univ Computer Science": "stem",
    "Primary Islamic Studies": "humanities",
    "Primary Geography": "social_science",
    "High Philosophy": "humanities",
    "Middle Arabic Language": "language",
    "Middle Economics": "social_science",
    "Univ Management": "other",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_arabicmmlu_template_yaml")
    parser.add_argument("--save_prefix_path", default="arabicmmlu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        # description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"arabicmmlu_{category}",
            "task": f"arabicmmlu_{subject.lower().replace(' ', '_')}",
            "task_alias": subject,
            "dataset_name": subject,
            # "description": description,
        }

        file_save_path = (
            args.save_prefix_path
            + f"_{subject.lower().replace(' ', '_').replace('(', '').replace(')', '')}.yaml"
        )
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    arabicmmlu_subcategories = [f"arabicmmlu_{category}" for category in ALL_CATEGORIES]

    file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "arabicmmlu",
                "task": arabicmmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
