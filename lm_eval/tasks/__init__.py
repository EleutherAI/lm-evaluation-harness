import os
import yaml
from typing import List, Union, Dict

from lm_eval import utils
from lm_eval import prompts
from lm_eval.api.task import TaskConfig, Task, ConfigurableTask
from lm_eval.api.registry import (
    register_task,
    register_group,
    TASK_REGISTRY,
    GROUP_REGISTRY,
    ALL_TASKS,
)

import logging

eval_logger = logging.getLogger("lm-eval")

# import python tasks
from .squad import SQuAD2



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
        if config["group"] == config["task"]:
            raise ValueError("task and group name cannot be the same")
        elif type(config["group"]) == str:
            group_name = [config["group"]]
        else:
            group_name = config["group"]

        for group in group_name:
            register_group(group)(SubClass)

    return 0


def register_configurable_group(config: Dict[str, str], yaml_path: str = None) -> int:
    group = config["group"]
    all_task_list = config["task"]
    config_list = [task for task in all_task_list if type(task) != str]
    task_list = [task for task in all_task_list if type(task) == str]

    for task_config in config_list:
        task_config = utils.load_yaml_config(yaml_path, task_config)
        var_configs = check_prompt_config(
            {
                **task_config,
                **{"group": group},
            },
            yaml_path=os.path.dirname(yaml_path),
        )
        for config in var_configs:
            register_configurable_task(config)

    task_names = utils.pattern_match(task_list, ALL_TASKS)
    for task in task_names:
        if (task in TASK_REGISTRY) or (task in GROUP_REGISTRY):
            if group in GROUP_REGISTRY:
                GROUP_REGISTRY[group].append(task)
            else:
                GROUP_REGISTRY[group] = [task]
                ALL_TASKS.add(group)

    return 0


def check_prompt_config(
    config: Dict[str, str], yaml_path: str = None
) -> List[Dict[str, str]]:
    all_configs = []
    if "use_prompt" in config:
        prompt_list = prompts.load_prompt_list(
            use_prompt=config["use_prompt"],
            dataset_name=config["dataset_path"],
            subset_name=config["dataset_name"] if "dataset_name" in config else None,
            yaml_path=yaml_path,
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
                                prompt_variation.split("/")[-1]
                                if ".yaml" in prompt_variation
                                else prompt_variation,
                            ]
                        )
                    },
                    **{"output_type": "generate_until"},
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


def include_task_folder(task_dir: str, register_task: bool = True) -> None:
    """
    Calling this function
    """
    for root, subdirs, file_list in os.walk(task_dir):
        # if (subdirs == [] or subdirs == ["__pycache__"]) and (len(file_list) > 0):
        for f in file_list:
            if f.endswith(".yaml"):
                yaml_path = os.path.join(root, f)
                try:
                    config = utils.load_yaml_config(yaml_path)

                    if "task" not in config:
                        continue

                    all_configs = check_prompt_config(
                        config, yaml_path=os.path.dirname(yaml_path)
                    )
                    for config in all_configs:
                        if register_task:
                            if type(config["task"]) == str:
                                register_configurable_task(config)
                        else:
                            if type(config["task"]) == list:
                                register_configurable_group(config, yaml_path)
                except ModuleNotFoundError as e:
                    eval_logger.warning(
                        f"{yaml_path}: {e}. Config will not be added to registry."
                    )
                except Exception as error:
                    import traceback

                    eval_logger.debug(
                        "Failed to load config in\n"
                        f"                                 {yaml_path}\n"
                        "                                 Config will not be added to registry\n"
                        f"                                 Error: {error}\n"
                        f"                                 Traceback: {traceback.format_exc()}"
                    )
    return 0


def include_path(task_dir):
    include_task_folder(task_dir)
    # Register Benchmarks after all tasks have been added
    include_task_folder(task_dir, register_task=False)
    return 0


task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
include_path(task_dir)


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
