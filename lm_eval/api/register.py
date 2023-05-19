import os

task_registry = {}
group_registry = {}
task2func_index = {}
func2task_index = {}


def register_task(name):
    def wrapper(func):

        task_registry[name] = func
        func2task_index[func.__name__] = name
        task2func_index[name] = func.__name__
        return func

    return wrapper


def register_group(name):
    def wrapper(func):

        func_name = func2task_index[func.__name__]

        if name in group_registry:
            group_registry[name].append(func_name)
        else:
            group_registry[name] = [func_name]
        return func

    return wrapper
