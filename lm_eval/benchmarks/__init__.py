import os
import yaml

from lm_eval import utils
from lm_eval.tasks import register_configurable_task, check_prompt_config
from lm_eval.logger import eval_logger
from lm_eval.api.registry import (
    TASK_REGISTRY,
    GROUP_REGISTRY,
    ALL_TASKS,
)


def include_benchmarks(task_dir):

    for root, subdirs, file_list in os.walk(task_dir):
        if (subdirs == [] or "__pycache__" in subdirs) and (len(file_list) > 0):
            for f in file_list:
                if f.endswith(".yaml"):
                    try:
                        benchmark_path = os.path.join(root, f)

                        with open(benchmark_path, "rb") as file:
                            yaml_config = yaml.full_load(file)

                        if "prompts" in yaml_config:
                            continue  # Skip it

                        assert "group" in yaml_config
                        group = yaml_config["group"]
                        all_task_list = yaml_config["task"]
                        config_list = [
                            task for task in all_task_list if type(task) != str
                        ]
                        task_list = [
                            task for task in all_task_list if type(task) == str
                        ]

                        for task_config in config_list:
                            yaml_dir = os.path.dirname(benchmark_path)
                            task_config = utils.load_yaml_config(
                                yaml_config=task_config, yaml_dir=yaml_dir
                            )
                            if "use_prompt" in task_config:
                                if "yaml" in task_config["use_prompt"]:
                                    task_config["use_prompt"] = os.path.join(
                                        root, task_config["use_prompt"]
                                    )

                            var_configs = check_prompt_config(
                                {
                                    **task_config,
                                    **{"group": group},
                                }
                            )
                            for config in var_configs:
                                register_configurable_task(config)

                        task_names = utils.pattern_match(task_list, ALL_TASKS)
                        for task in task_names:
                            if task in TASK_REGISTRY:
                                if group in GROUP_REGISTRY:
                                    GROUP_REGISTRY[group].append(task)
                                else:
                                    GROUP_REGISTRY[group] = [task]
                                    ALL_TASKS.add(group)
                    except Exception as error:
                        eval_logger.warning(
                            "Failed to load benchmark in\n"
                            f"                                 {benchmark_path}\n"
                            "                                 Benchmark will not be added to registry\n"
                            f"                                 Error: {error}"
                        )


task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
include_benchmarks(task_dir)
