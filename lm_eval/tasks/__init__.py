import os
from typing import List, Union

from lm_eval import utils
from lm_eval.logger import eval_logger
from lm_eval.api.task import TaskConfig, Task, ConfigurableTask
from lm_eval.api.registry import (
    register_task,
    register_group,
    TASK_REGISTRY,
    GROUP_REGISTRY,
    ALL_TASKS,
)


def get_task_name_from_config(task_config):
    return "{dataset_path}_{dataset_name}".format(**task_config)


def include_task_folder(task_dir):
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

                        SubClass = type(
                            config["task"] + "ConfigurableTask",
                            (ConfigurableTask,),
                            {"CONFIG": TaskConfig(**config)},
                        )

                        if "task" in config:
                            # task_name = "{}:{}".format(
                            #     get_task_name_from_config(config), config["task"]
                            # )
                            task_name = "{}".format(config["task"])
                            register_task(task_name)(SubClass)

                        if "group" in config:
                            for group in config["group"]:
                                register_group(group)(SubClass)
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
def get_task_dict(task_name_list: List[Union[str, dict, Task]], **kwargs):

    config = {**kwargs}

    task_name_from_registry_dict = {}
    task_name_from_config_dict = {}
    task_name_from_object_dict = {}

    for task_element in task_name_list:
        if isinstance(task_element, str):

            if task_element in GROUP_REGISTRY:
                for task_name in GROUP_REGISTRY[task_element]:
                    if task_name not in task_name_from_registry_dict:
                        task_name_from_registry_dict = {
                            **task_name_from_registry_dict,
                            task_name: get_task(task_name=task_name, config=config),
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

    # task_name_from_registry_dict = {
    #     task_name: get_task(
    #         task_name=task_name,
    #         task_config=config
    #     )
    #     for group_name in task_name_list for task_name in GROUP_REGISTRY[group_name]
    #     if (isinstance(group_name, str)) and (group_name in GROUP_REGISTRY)
    # }
    # task_name_from_config_dict = {
    #     get_task_name_from_config(task_config): ConfigurableTask(
    #         config=task_config
    #     )
    #     for task_config in task_name_list
    #     if isinstance(task_config, dict)
    # }
    # # TODO: Do we still need this?
    # task_name_from_object_dict = {
    #     get_task_name_from_object(task_object): task_object
    #     for task_object in task_name_list
    #     if isinstance(task_object, Task)
    # }

    assert set(task_name_from_registry_dict.keys()).isdisjoint(
        set(task_name_from_object_dict.keys())
    )
    return {
        **task_name_from_registry_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }
