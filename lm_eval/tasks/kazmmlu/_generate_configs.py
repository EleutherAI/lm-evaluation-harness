"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm

import yaml

def function_constructor(loader, node):
    value = loader.construct_scalar(node)
    # This will return the function path as a string
    return value

yaml.add_constructor('!function', function_constructor)

eval_logger = logging.getLogger("lm-eval")


SUBJECTS = {
    'Accounting and Auditing (Professional & University in rus)': 'social_science',
    'Biology (High School in kaz)': 'stem',
    'Biology (High School in rus)': 'stem',
    'Biology (Professional & University in rus)': 'stem',
    'Chemistry (High School in kaz)': 'stem',
    'Chemistry (High School in rus)': 'stem',
    'Culture and Art (Professional & University in rus)': 'humanities',
    'Economics and Entrepreneurship (Professional in rus)': 'social_science',
    'Education and Training (Professional & University in rus)': 'social_science',
    'Finance (Professional & University in rus)': 'social_science',
    'General Education Disciplines (Professional & University in rus)': 'other',
    'Geography (High School in kaz)': 'social_science',
    'Geography (High School in rus)': 'social_science',
    'Informatics (High School in kaz)': 'stem',
    'Informatics (High School in rus)': 'stem',
    'Jurisprudence (Professional & University in rus)': 'social_science',
    'Kazakh History (High School in kaz)': 'humanities',
    'Kazakh History (High School in rus)': 'humanities',
    'Kazakh Language (High School in kaz)': 'language',
    'Kazakh Literature (High School in kaz)': 'language',
    'Law (High School in kaz)': 'social_science',
    'Law (High School in rus)': 'social_science',
    'Management and Marketing (Professional & University in rus)': 'social_science',
    'Math (High School in kaz)': 'stem',
    'Math (High School in rus)': 'stem',
    'Math Literacy (High School in rus)': 'stem',
    'Medicine (Professional & University in rus)': 'stem',
    'Philosophy and Psychology (Professional & University in rus)': 'humanities',
    'Physics (High School in kaz)': 'stem',
    'Physics (High School in rus)': 'stem',
    'Reading Literacy (High School in kaz)': 'humanities',
    'Reading Literacy (High School in rus)': 'humanities',
    'Russian Language (High School in rus)': 'language',
    'Russian Literature (High School in rus)': 'language',
    'Social Science (Professional & University in rus)': 'social_science',
    'World History (High School in kaz)': 'humanities',
    'World History (High School in rus)': 'humanities'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", default="_default_kazmmlu_template_yaml")
    parser.add_argument("--save_prefix_path", default="kazmmlu")
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
            "tag": f"kazmmlu_{category}",
            "task": f"kazmmlu_{subject.lower().replace(' ', '_')}",
            "task_alias": subject,
            "dataset_name": subject,
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

    kazmmlu_subcategories = [f"kazmmlu_{category}" for category in ALL_CATEGORIES]

    file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "kazmmlu",
                "task": kazmmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )