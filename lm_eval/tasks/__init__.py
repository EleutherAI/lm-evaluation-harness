import os
import re

from typing import List, Union

from .vanilla import *

from lm_eval.utils import get_yaml_config, register_task
from lm_eval.api.task import Task, ConfigurableTask

YAML_REGISTRY = {}
FUNC_REGISTRY = register_task.all
BENCHMARK_REGISTRY = {}

# we want to register all yaml tasks in our .yaml folder.
yaml_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + "yaml"
for yaml_file in sorted(os.listdir(yaml_dir)):
    yaml_path = os.path.join(yaml_dir, yaml_file)

    names = re.sub(r"\.", "_", yaml_path.split("/")[-1])
    YAML_REGISTRY[names] = yaml_path

TASK_REGISTRY = list(YAML_REGISTRY.keys()) + list(FUNC_REGISTRY.keys())
ALL_TASKS = sorted(list(TASK_REGISTRY))


def get_task(task_name, task_config):

    if task_name in TASK_REGISTRY:
        if task_name in YAML_REGISTRY:
            return ConfigurableTask(
                config={
                    **get_yaml_config(YAML_REGISTRY[task_name]),
                    **task_config
                }
            )
        elif task_name in FUNC_REGISTRY:
            return FUNC_REGISTRY[task_name](
                config=task_config
            )
    else:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
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


def get_task_name_from_config(task_config):
    return "configurable_{dataset_path}_{dataset_name}".format(**task_config)


# TODO: pass num_fewshot and other cmdline overrides in a better way
def get_task_dict(task_name_list: List[Union[str, dict, Task]], num_fewshot=None):
    task_name_from_registry_dict = {
        task_name: get_task(
            task_name=task_name,
            task_config={"num_fewshot": num_fewshot if num_fewshot else 0}
        )
        for task_name in task_name_list
        if isinstance(task_name, str)
    }
    task_name_from_config_dict = {
        get_task_name_from_config(task_config): ConfigurableTask(
            config=task_config
        )
        for task_config in task_name_list
        if isinstance(task_config, dict)
    }
    # TODO: Do we still need this?
    # task_name_from_object_dict = {
    #     get_task_name_from_object(task_object): task_object
    #     for task_object in task_name_list
    #     if isinstance(task_object, Task)
    # }
    # assert set(task_name_from_registry_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {
        **task_name_from_registry_dict,
        **task_name_from_config_dict,
        # **task_name_from_object_dict,
    }


