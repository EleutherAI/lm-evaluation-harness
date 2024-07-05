import json
import os
from typing import Dict, List


def filter_dict(d, key) -> Dict:
    return {k: v for k, v in d.items() if key in k}


class ParseConfig:
    def __init__(self, dict_obj):
        for key, value in dict_obj.items():
            if isinstance(value, dict):
                setattr(self, key, ParseConfig(value))
            elif isinstance(value, list):
                setattr(self, key, [ParseConfig(i) for i in value])
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        raise AttributeError(f"Attribute '{name}' is not defined in the config file.")

    def __getitem__(self, key):
        return self.__dict__[key]

    def to_dict(self):
        return self.__dict__

    def keys(self):
        if hasattr(self, "__dict__"):
            return self.__dict__.keys()
        else:
            raise TypeError("Object is not a dictionary.")

    def items(self):
        if hasattr(self, "__dict__"):
            return iter(self.__dict__.items())
        else:
            raise TypeError("Object is not a dictionary.")


def load_all_configs(device: str) -> List[Dict]:
    """
    Given a device, load all the config files for that device.

    Args:
        device (str): The device to load the config files for. If None, will use "cpu" as default.
    """
    if device is None:
        device = "cpu"
    if device not in ["cpu", "gpu"]:
        raise ValueError(f"Invalid device {device}. Must be one of ['cpu', 'gpu']")
    data_list = []
    configs_dir = "tests/leaderboards/testconfigs"
    for filename in os.listdir(configs_dir):
        if device in filename and filename.endswith(".json"):
            with open(f"{configs_dir}/{filename}", "r") as f:
                data_list.append(json.load(f))
    return data_list
