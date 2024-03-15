import os
import yaml
import argparse

from lm_eval.utils import logging


LANGS = [
    "BG",
    "DA",
    "DE",
    "ET",
    "FI",
    "FR",
    "EL",
    "IT",
    "LV",
    "LT",
    "NL",
    "PL",
    "PT-PT",
    "RO",
    "SV",
    "SK",
    "SL",
    "ES",
    "CS",
    "HU",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_yaml_path", required=True)
    parser.add_argument("--save_prefix_path", default="ogx_hellaswagx")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]

    for lang in LANGS:
        yaml_dict = {
            "include": base_yaml_name,
            "dataset_name": lang,
            "task": f"ogx_hellaswagx_{lang.lower()}",
        }

        file_save_path = args.save_prefix_path + f"_{lang.lower()}.yaml"

        logging.info(f"Saving yaml for subset {lang} to {file_save_path}")

        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
                sort_keys=False,
            )
