"""
Take in a YAML, and output all other splits with this YAML
"""
import argparse
import os

from datasets import get_dataset_config_names
import yaml
from tqdm import tqdm

from lm_eval.utils import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="sib200")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our other YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    with open(args.base_yaml_path, encoding="utf-8") as f:
        base_yaml = yaml.full_load(f)        

    for lang in tqdm(get_dataset_config_names("wikimedia/wikipedia")):
        yaml_dict = {
            "include": base_yaml_name,
            "task": f"wiki_ppl_{lang}",
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
