import argparse
import logging
import os

import yaml
from promptsource.templates import DatasetTemplates
from tqdm import tqdm


# from lm_eval.api.registry import ALL_TASKS
eval_logger = logging.getLogger(__name__)


# from lm_eval.tasks import include_task_folder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", required=True)
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--task_save_path", default="lm_eval/tasks/")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.benchmark_path, encoding="utf-8") as file:
        TASK_LIST = yaml.full_load(file)
        for task in tqdm(TASK_LIST):
            eval_logger.info(f"Processing {task}")

            dataset_name = task["dataset_path"]
            if "dataset_name" in task:
                subset_name = task["dataset_name"]
                file_subdir = f"{dataset_name}/{subset_name}"
            else:
                subset_name = None
                file_subdir = f"{dataset_name}"

            file_path = os.path.join(args.task_save_path, file_subdir, "promptsource/")

            os.makedirs(file_path, exist_ok=True)

            if subset_name is None:
                prompts = DatasetTemplates(dataset_name=dataset_name)
            else:
                prompts = DatasetTemplates(
                    dataset_name=dataset_name, subset_name=subset_name
                )

            for idx, prompt_name in enumerate(prompts.all_template_names):
                full_file_name = f"promptsource_{idx}.yaml"
                config_dict = {
                    "group": args.benchmark_name,
                    "include": "promptsource_template.yaml",
                    "use_prompts": f"promptsource:{prompt_name}",
                }

                file_save_path = os.path.join(file_path, full_file_name)
                eval_logger.info(f"Save to {file_save_path}")
                with open(file_save_path, "w", encoding="utf-8") as yaml_file:
                    yaml.dump(config_dict, yaml_file)
