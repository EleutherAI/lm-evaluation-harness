import json
from typing import List
from lm_eval.utils import load_yaml_config
from pathlib import Path


FILE_PATH = file_path = ".github/outputs/tasks_all_changed_and_modified_files.txt"


def load_changed_files(file_path: str = FILE_PATH) -> List[str]:
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def parser(full_path: List[str]) -> List[str]:
    _output = set()
    for x in full_path:
        if x.endswith(".yaml"):
            _output.add(load_yaml_config(x)["task"])
        elif x.endswith(".py"):
            path = [str(x) for x in (list(Path(x).parent.glob("*.yaml")))]
            _output |= {load_yaml_config(x)["task"] for x in path}
    return list(_output)
