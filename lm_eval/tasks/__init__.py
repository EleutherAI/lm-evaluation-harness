import os
import yaml
from typing import List, Union, Dict

from lm_eval import utils
from lm_eval import prompts
from lm_eval.logger import eval_logger
from lm_eval.api.task import TaskConfig, Task, ConfigurableTask
from lm_eval.api.registry import (
    register_task,
    register_group,
    TASK_REGISTRY,
    GROUP_REGISTRY,
    ALL_TASKS,
)


def register_configurable_task(config: Dict[str, str]) -> int:
    SubClass = type(
        config["task"] + "ConfigurableTask",
        (ConfigurableTask,),
        {"CONFIG": TaskConfig(**config)},
    )

    if "task" in config:
        task_name = "{}".format(config["task"])
        register_task(task_name)(SubClass)

    if "group" in config:
        if type(config["group"]) == str:
            group_name = [config["group"]]
        else:
            group_name = config["group"]

        for group in group_name:
            register_group(group)(SubClass)

    return 0


def check_prompt_config(config: Dict[str, str]) -> List[Dict[str, str]]:
    all_configs = []
    if "use_prompt" in config:
        prompt_list = prompts.load_prompt_list(
            use_prompt=config["use_prompt"],
            dataset_name=config["dataset_path"],
            subset_name=config["dataset_name"] if "dataset_name" in config else None,
        )
        for idx, prompt_variation in enumerate(prompt_list):
            all_configs.append(
                {
                    **config,
                    **{"use_prompt": prompt_variation},
                    **{
                        "task": "_".join(
                            [
                                config["task"]
                                if "task" in config
                                else get_task_name_from_config(config),
                                prompt_variation,
                            ]
                        )
                    },
                    **{"output_type": "greedy_until"},
                }
            )
    else:
        all_configs.append(config)
    return all_configs


def get_task_name_from_config(task_config: Dict[str, str]) -> str:
    if "dataset_name" in task_config:
        return "{dataset_path}_{dataset_name}".format(**task_config)
    else:
        return "{dataset_path}".format(**task_config)


def include_task_folder(task_dir: str) -> None:
    """
    Calling this function
    """
    for root, subdirs, file_list in os.walk(task_dir):
        if (subdirs == [] or subdirs == ["__pycache__"]) and (len(file_list) > 0):
            for f in file_list:
                if f.endswith(".yaml"):
                    yaml_path = os.path.join(root, f)
                    try:
                        config = utils.load_yaml_config(yaml_path)
                        all_configs = check_prompt_config(config)
                        for config in all_configs:
                            register_configurable_task(config)

                    except Exception as error:
                        eval_logger.warning(
                            "Failed to load config in\n"
                            f"                                 {yaml_path}\n"
                            "                                 Config will not be added to registry\n"
                            f"                                 Error: {error}"
                        )


task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
include_task_folder(task_dir)


def get_task(task_name, config):
    try:
        return TASK_REGISTRY[task_name](config=config)
    except KeyError:
        eval_logger.info("Available tasks:")
        eval_logger.info(list(TASK_REGISTRY) + list(GROUP_REGISTRY))
        raise KeyError(f"Missing task {task_name}")


def get_task_name_from_object(task_object):
    for name, class_ in TASK_REGISTRY.items():
        if class_ is task_object:
            return name

    # TODO: scrap this
    # this gives a mechanism for non-registered tasks to have a custom name anyways when reporting
    return (
        task_object.EVAL_HARNESS_NAME
        if hasattr(task_object, "EVAL_HARNESS_NAME")
        else type(task_object).__name__
    )


# TODO: pass num_fewshot and other cmdline overrides in a better way
def get_task_dict(task_name_list: List[Union[str, Dict, Task]], **kwargs):

    config = {**kwargs}

    task_name_from_registry_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    if type(task_name_list) != list:
        task_name_list = [task_name_list]

    for task_element in task_name_list:
        if isinstance(task_element, str):

            if task_element in GROUP_REGISTRY:
                group_name = task_element
                for task_name in GROUP_REGISTRY[task_element]:
                    if task_name not in task_name_from_registry_dict:
                        task_obj = get_task_dict(task_name)
                        if task_name in task_obj.keys():
                            task_dict = {
                                task_name: (group_name, task_obj[task_name]),
                            }
                        else:
                            task_dict = {
                                task_name: (group_name, None),
                                **task_obj,
                            }

                        task_name_from_registry_dict = {
                            **task_name_from_registry_dict,
                            **task_dict,
                        }
            else:
                task_name = task_element
                if task_name not in task_name_from_registry_dict:
                    task_name_from_registry_dict = {
                        **task_name_from_registry_dict,
                        task_name: get_task(task_name=task_element, config=config),
                    }

        elif isinstance(task_element, dict):
            task_element.update(config)
            task_name_from_config_dict = {
                **task_name_from_config_dict,
                get_task_name_from_config(task_element): ConfigurableTask(
                    config=task_element
                ),
            }

        elif isinstance(task_element, Task):

            task_name_from_object_dict = {
                **task_name_from_object_dict,
                get_task_name_from_object(task_element): task_element,
            }

    assert set(task_name_from_registry_dict.keys()).isdisjoint(
        set(task_name_from_object_dict.keys())
    )
    return {
        **task_name_from_registry_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }
