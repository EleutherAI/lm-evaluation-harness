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
    "Islamic Studies": "humanities",
    "Driving Test": "other",
    "Natural Science (Middle School)": "stem",
    "Natural Science (Primary School)": "stem",
    "History (Primary School)": "humanities",
    "History (Middle School)": "humanities",
    "History (High School)": "humanities",
    "General Knowledge": "other",
    "General Knowledge (Primary School)": "other",
    "General Knowledge (Middle School)": "other",
    "Law (Professional)": "humanities",
    "Physics (High School)": "stem",
    "Social Science (Middle School)": "social_science",
    "Social Science (Primary School)": "social_science",
    "Management (University)": "other",
    "Arabic Language (Primary School)": "language",
    "Arabic Language (Middle School)": "language",
    "Arabic Language (High School)": "language",
    "Political Science (University)": "social_science",
    "Philosophy (High School)": "humanities",
    "Accounting (University)": "social_science",
    "Computer Science (University)": "stem",
    "Computer Science (Middle School)": "stem",
    "Computer Science (Primary School)": "stem",
    "Computer Science (High School)": "stem",
    "Geography (Primary School)": "social_science",
    "Geography (Middle School)": "social_science",
    "Geography (High School)": "social_science",
    "Math (Primary School)": "stem",
    "Biology (High School)": "stem",
    "Economics (University)": "social_science",
    "Economics (Middle School)": "social_science",
    "Economics (High School)": "social_science",
    "Arabic Language (General)": "language",
    "Arabic Language (Grammar)": "language",
    "Islamic Studies (High School)": "humanities",
    "Islamic Studies (Middle School)": "humanities",
    "Islamic Studies (Primary School)": "humanities",
    "Civics (Middle School)": "social_science",
    "Civics (High School)": "social_science",
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

    # with open(args.base_yaml_path, encoding="utf-8") as f:
    #     base_yaml = yaml.full_load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        # description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"arabicmmlu_{category}_tasks",
            "task": f"arabicmmlu_{subject.lower().replace(' ', '_').replace('(', '').replace(')', '')}",
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
