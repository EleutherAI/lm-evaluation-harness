"""
Take in a YAML, and output all other splits with this YAML
"""
import os
import yaml
import argparse
import requests

from tqdm import tqdm

from lm_eval.logger import eval_logger

API_URL = "https://datasets-server.huggingface.co/splits?dataset=facebook/belebele"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="belebele")
    parser.add_argument("--cot_prompt_path", default=None)
    parser.add_argument("--task_prefix", default="")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path) as f:
        base_yaml = yaml.full_load(f)

    if args.cot_prompt_path is not None:
        import json

        with open(args.cot_prompt_path) as f:
            cot_file = json.load(f)

    def query():
        response = requests.get(API_URL)
        return response.json()["splits"]

    languages = [split["config"] for split in query()]

    for lang in tqdm(languages):
        yaml_dict = {
            "include": base_yaml_name,
            "task": f"belebele_{args.task_prefix}_{lang}"
            if args.task_prefix != ""
            else f"belebele_{lang}",
            "dataset_name": lang,
        }

        file_save_path = args.save_prefix_path + f"_{lang}.yaml"
        eval_logger.info(f"Saving yaml for subset {lang} to {file_save_path}")
        with open(file_save_path, "w") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
