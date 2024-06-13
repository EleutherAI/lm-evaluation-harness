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
    "business": "other",
    "law": "humanities",
    "psychology": "social_sciences",
    "biology": "stem",
    "chemistry": "stem",
    "history": "humanities",
    "other": "other",
    "health": "other",
    "economics": "social_sciences",
    "math": "stem",
    "physics": "stem",
    "computer_science": "stem",
    "philosophy": "humanities",
    "engineering": "stem"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="mmlu_pro")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    parser.add_argument("--group_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path, encoding="utf-8") as f:
            cot_file = json.load(f)

    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        if args.cot_prompt_path is not None:
            description = cot_file[subject]
        else:
            description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "group": f"mmlu_pro_{args.task_prefix}_{category}"
            if args.task_prefix != ""
            else f"mmlu_pro_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"mmlu_pro_{args.task_prefix}_{subject}"
            if args.task_prefix != ""
            else f"mmlu_pro_{subject}",
            "task_alias": subject.replace("_", " "),
            "dataset_name": subject,
            "description": description,
        }

        file_save_path = args.save_prefix_path + f"_{subject}.yaml"
        eval_logger.info(f"Saving yaml for subset {subject} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    if args.task_prefix != "":
        mmlu_pro_subcategories = [
            f"mmlu_pro_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        mmlu_pro_subcategories = [f"mmlu_pro_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"mmlu_pro_{args.task_prefix}"
                if args.task_prefix != ""
                else "mmlu_pro",
                "task": mmlu_pro_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
