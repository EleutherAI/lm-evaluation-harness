"""
Take in a YAML, and output all other splits with this YAML
"""
import os
import re
import yaml
import requests
import argparse

import datasets
from tqdm import tqdm

from lm_eval import utils
from lm_eval.logger import eval_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="flan_zeroshot")
    parser.add_argument("--cot", default=False)
    parser.add_argument("--fewshot", default=False)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path) as f:
        base_yaml = yaml.full_load(f)

    base_doc_to_text = "Q: {{input}}\nA:"
    answer_regex = re.compile("(?<=answer is )(.*)(?=.)")

    dataset_path = "lukaemon/bbh"
    for task in tqdm(datasets.get_dataset_infos(dataset_path).keys()):

        resp = requests.get(
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{task}.txt"
        ).content.decode("utf-8")
        prompt = resp.split("\n-----\n")[-1]
        description, *few_shot = prompt.split("\n\nQ:")

        prefix_doc_to_text = ""
        if args.fewshot:
            if args.cot:
                prefix_doc_to_text = " ".join(few_shot)
            else:
                for shot in few_shot:
                    shot = "Q:" + shot
                    try:
                        answer = answer_regex.search(shot)[0]
                    except Exception:
                        print("task", task)
                        print(shot)
                    example = shot.split("Let's think step by step.")[0]
                    prefix_doc_to_text += f"{example}{answer}\n\n"

        doc_to_text = prefix_doc_to_text + base_doc_to_text
        if args.cot:
            doc_to_text = doc_to_text + " Let's think step by step.\n"

        yaml_dict = {
            "include": base_yaml_name,
            "task": f"bbh_{args.task_prefix}_{task}",
            "dataset_name": task,
            "description": description + "\n\n",
            "doc_to_text": doc_to_text,
        }

        file_save_path = args.save_prefix_path + f"/{task}.yaml"
        eval_logger.info(f"Saving yaml for subset {task} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
