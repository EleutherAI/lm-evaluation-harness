import os
import re
import yaml
from typing import List, Union

from .arc import *

from lm_eval.api.task import TaskConfig, Task, ConfigurableTask
from lm_eval.api.register import (
    register_task,
    register_group,
    task_registry,
    group_registry
    )

task_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
for root, subdirs, file_list in os.walk(task_dir):
    if (subdirs == []) and (len(file_list) > 0):
        for file in file_list:
            if "yaml" in file:
                yaml_path = os.path.join(root, file)
                try:
                    config = yaml.full_load(open(yaml_path, "rb"))

                    SubClass = type(
                        config['task']+'ConfigurableTask',
                        (ConfigurableTask,),
                        {'CONFIG': TaskConfig(**config)}
                    )

                    if 'task' in config:
                        register_task(config['task'])(SubClass)
                    
                    if 'group' in config:
                        for group in config['group']:
                            register_group(group)(SubClass)
                except:
                    pass        

TASK_REGISTRY = task_registry
GROUP_REGISTRY = group_registry
ALL_TASKS = sorted(list(TASK_REGISTRY))

def get_task(task_name, config):
    try:
        return TASK_REGISTRY[task_name](config)
    except KeyError:
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
def get_task_dict(task_name_list: List[Union[str, dict, Task]], config, **kwargs):

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
                            task_name: get_task(
                                task_name=task_name, config=config
                                )
                            }
            else:
                if task_name not in task_name_from_registry_dict:
                    task_name_from_registry_dict = {
                        **task_name_from_registry_dict,
                        task_name: get_task(
                            task_name=task_element, config=config
                            )
                        }

        elif isinstance(task_element, dict):

            task_name_from_config_dict = {
                **task_name_from_config_dict,
                get_task_name_from_config(task_element): ConfigurableTask(
                    config=config
                )
            }

        elif isinstance(task_element, Task):

            task_name_from_object_dict = {
                **task_name_from_object_dict,
                get_task_name_from_object(task_element): task_element
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

    assert set(task_name_from_registry_dict.keys()).isdisjoint(set(task_name_from_object_dict.keys()))
    return {
        **task_name_from_registry_dict,
        **task_name_from_config_dict,
        **task_name_from_object_dict,
    }


