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


metric_registry = {}
aggregation_registry = {}
default_aggregation_registry = {}
higher_is_better_registry = {}
output_type_registry = {}
metric2func_index = {}
func2metric_index = {}
aggregation2func_index = {}
func2aggregation_index = {}


def register_metric(name):
    def wrapper(func):

        metric_registry[name] = func
        func2metric_index[func.__name__] = name
        metric2func_index[name] = func.__name__
        return func

    return wrapper


def register_aggregation(name):
    def wrapper(func):

        aggregation_registry[name] = func
        func2aggregation_index[func.__name__] = name
        aggregation2func_index[name] = func.__name__
        return func

    return wrapper


def register_default_aggregation(aggregation):
    def wrapper(func):

        if aggregation in aggregation_registry:
            metric_name = func2metric_index[func.__name__]
            default_aggregation_registry[metric_name] = aggregation
        else:
            print("aggregation not registered")
        return func

    return wrapper


def register_higher_is_better(higher_is_better):
    def wrapper(func):

        if func.__name__ in func2metric_index:
            metric_name = func2metric_index[func.__name__]
            higher_is_better_registry[metric_name] = higher_is_better
        else:
            pass

        return func

    return wrapper


def register_output_type(output_type):
    def wrapper(func):

        if func.__name__ in func2metric_index:
            metric_name = func2metric_index[func.__name__]
            output_type_registry[metric_name] = output_type
        else:
            pass

        return func

    return wrapper
