import os
import argparse

from lm_eval import utils
from promptsource.templates import DatasetTemplates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model_args", default="")


def main():
    args = parse_args()

    path = args.benchmark
    yaml_path = ""
    with open(path) as file:
        TASK_LIST = file.readlines()
        for dataset_name, subset_name in TASK_LIST:

            if subset_name is None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(
                    dataset_name=dataset_name, subset_name=subset_name
                )

            with open(os.path.join(yaml_path, "promptsource_template.yaml")) as file:
                yaml_dict = file.readline()

            for prompt_name in prompts.all_template_names:
                config_dict = {
                    "include": "promptsource_template.yaml",
                    "use_prompts": prompts[prompt_name],
                    **yaml_dict,
                }

                return config_dict
