"""
Take in a YAML, and output all "other" splits with this YAML
"""

import argparse
import logging
import os

import yaml
from tqdm import tqdm


eval_logger = logging.getLogger("lm-eval")

countries = {
    "KSA": "Gulf",
    "UAE": "Gulf",
    "Yemen": "Gulf",
    "Lebanon": "Levant",
    "Syria": "Levant",
    "Palestine": "Levant",
    "Jordan": "Levant",
    "Tunisia": "North Africa",
    "Algeria": "North Africa",
    "Morocco": "North Africa",
    "Libya": "North Africa",
    "Egypt": "Nile Valley",
    "Sudan": "Nile Valley",
}

VERSION = 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_yaml_path", default="_default_arab_culture_completion_template_yaml"
    )
    parser.add_argument("--save_prefix_path", default="arab_culture_completion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # get filename of base_yaml so we can `"include": ` it in our "other" YAMLs.
    base_yaml_name = os.path.split(args.base_yaml_path)[-1]
    # with open(args.base_yaml_path, encoding="utf-8") as f:
    #     base_yaml = yaml.full_load(f)

    ALL_REGIONS = []
    for country, region in tqdm(countries.items()):
        if region not in ALL_REGIONS:
            ALL_REGIONS.append(region)

        # description = f"The following are multiple choice questions (with answers) about {' '.join(subject.split('_'))}.\n\n"

        yaml_dict = {
            "include": base_yaml_name,
            "tag": f"arab_culture_completion_{region.lower().replace(' ', '_')}_tasks",
            "task": f"arab_culture_completion_{country.lower().replace(' ', '_')}",
            "task_alias": country,
            "dataset_name": country,
            # "description": description,
        }

        file_save_path = (
            args.save_prefix_path
            + f"_{country.lower().replace(' ', '_').replace('(', '').replace(')', '')}.yaml"
        )
        eval_logger.info(f"Saving yaml for subset {country} to {file_save_path}")
        with open(file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                yaml_dict,
                yaml_file,
                allow_unicode=True,
                default_style='"',
            )

    arab_culture_completion_regions = [
        f"arab_culture_completion_{region.lower().replace(' ', '_')}"
        for region in ALL_REGIONS
    ]

    file_save_path = args.save_prefix_path + ".yaml"

    eval_logger.info(f"Saving benchmark config to {file_save_path}")

    for region in ALL_REGIONS:
        file_save_path = (
            args.save_prefix_path + f"_{region.lower().replace(' ', '_')}.yaml"
        )
        eval_logger.info(f"Saving yaml for subset {region} to {file_save_path}")
        with open("_" + file_save_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(
                {
                    "group": f"arab_culture_completion_{region.lower().replace(' ', '_')}",
                    "group_alias": region,
                    "task": [
                        f"arab_culture_completion_{region.lower().replace(' ', '_')}_tasks"
                    ],
                    "aggregate_metric_list": {"metric": "acc", "weight_by_size": True},
                    "metadata": {
                        "description": "arab Culture tasks",
                        "version": VERSION,
                    },
                },
                yaml_file,
                indent=4,
                default_flow_style=False,
            )

    file_save_path = args.save_prefix_path + ".yaml"
    with open("_" + file_save_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(
            {
                "group": "arab_culture_completion",
                "task": arab_culture_completion_regions,
                "aggregate_metric_list": {"metric": "acc", "weight_by_size": True},
                "metadata": {"description": "Arab Culture tasks", "version": VERSION},
            },
            yaml_file,
            indent=4,
            default_flow_style=False,
        )
