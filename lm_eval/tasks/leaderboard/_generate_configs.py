"""
Take in a YAML, and output all other splits with this YAML
"""
import argparse
import os
import re

import datasets
import requests
import yaml
from tqdm import tqdm

from lm_eval import utils
from lm_eval.tasks.mmlu._generate_configs import SUBJECTS
eval_logger = logging.getLogger("lm-eval")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="zeroshot")
    parser.add_argument("--cot", default=False)
    parser.add_argument("--fewshot", default=False)
    return parser.parse_args()

def leaderboard_bbh_config(args):
    base_doc_to_text = "Q: {{input}}\nA:"

    dataset_path = "lukaemon/leaderboard_bbh"
    for task in tqdm(datasets.get_dataset_infos(dataset_path).keys()):
        resp = requests.get(
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{task}.txt"
        ).content.decode("utf-8")

        prompt = resp.split("\n-----\n")[-1]
        description, _ = prompt.split("\n\n")
        prefix_doc_to_text = ""
        doc_to_text = prefix_doc_to_text + base_doc_to_text

        yaml_dict = {
            "include": args.base_yaml_name,
            "task": f"leaderboard_bbh_{task}",
            "dataset_name": task,
            "description": description + "\n\n",
            "doc_to_text": doc_to_text,
        }

        file_save_path = args.save_prefix_path + f"/{task}.yaml"
        utils.eval_logger.info(f"Saving yaml for subset {task} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )

def leaderboard_mmlu_config(args):
    ALL_CATEGORIES = []
    for subject, category in tqdm(SUBJECTS.items()):
        if category not in ALL_CATEGORIES:
            ALL_CATEGORIES.append(category)

        description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": args.base_yaml_name,
            "group": f"leaderboard_mmlu_{category}",
            "group_alias": category.replace("_", " "),
            "task": f"leaderboard_mmlu_{subject}",
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
        leaderboard_mmlu_subcategories = [
            f"leaderboard_mmlu_{args.task_prefix}_{category}" for category in ALL_CATEGORIES
        ]
    else:
        leaderboard_mmlu_subcategories = [f"leaderboard_mmlu_{category}" for category in ALL_CATEGORIES]

    if args.group_prefix != "":
        file_save_path = args.group_prefix + ".yaml"
    else:
        file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")
    with open(file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": f"leaderboard_mmlu_{args.task_prefix}"
                if args.task_prefix != ""
                else "leaderboard_mmlu",
                "task": leaderboard_mmlu_subcategories,
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    args.base_yaml_name = os.path.split(args.base_yaml_path)[-1]


    leaderboard_bbh_config(args)
    leaderboard_mmlu_config(args)

