"""
Take in a YAML, and output all other splits with this YAML
"""
import argparse
import os

import requests
import yaml
from tqdm import tqdm

from lm_eval.utils import logging


API_URL = "https://datasets-server.huggingface.co/splits?dataset=mhardalov/exams"
CONFIGS_TO_IGNORE = ['aignments', 'multilingual', 'multilingual_with_para']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="sib200")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    breakpoint()
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)

    def query():
        response = requests.get(API_URL)
        return response.json()["splits"]
    
    languages = []
    for lang_config in query():
        if lang_config['config'] not in CONFIGS_TO_IGNORE and lang_config['split'] == 'validation' and 'para' not in lang_config['config']:
            languages.append(lang_config["config"])
    
    for lang in tqdm([lang for lang in languages]):
        yaml_dict = {
            "include": base_yaml_name,
            "task": f"exams_closed_book_{lang}",
            "test_split": 'validation',
            "fewshot_split": 'validation',
            "dataset_name": f"{lang}"
        }

        file_save_path = args.save_prefix_path + f"_{lang}.yaml"
        logging.info(f"Saving yaml for subset {lang} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                width=float("inf"),
                allow_unicode=True,
                default_style='"',
            )
