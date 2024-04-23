import numpy as np


def sanitize_numpy(example_dict):
    output_dict = {}
    for k, v in example_dict.items():
        if isinstance(v, np.generic):
            output_dict[k] = v.item()
        else:
            output_dict[k] = v
    return output_dict


def as_list(item):
    if isinstance(item, list):
        return item
    elif isinstance(item, tuple):
        return list(item)
    return [item]
