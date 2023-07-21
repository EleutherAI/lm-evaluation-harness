import json
from typing import List
from lm_eval.utils import load_yaml_config
from pathlib import Path
import sys

# This is the path where the output for the changed files for the tasks folder is stored
# FILE_PATH = file_path = ".github/outputs/tasks_all_changed_and_modified_files.txt"


# reads a text file and returns a list of words
# used to read the output of the changed txt from tj-actions/changed-files
def load_changed_files(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        content = f.read()
        words_list = [x for x in content.split()]
        sys.stdout.write(f"list of files: {words_list}")
    return words_list


# checks the txt file for list of changed files.
# if file ends with .yaml then check yaml for task name
# if file ends with .py then parse the folder for all yaml files
def parser(full_path: List[str]) -> List[str]:
    _output = set()
    for x in full_path:
        if x.endswith(".yaml"):
            _output.add(load_yaml_config(x)["task"])
        elif x.endswith(".py"):
            path = [str(x) for x in (list(Path(x).parent.glob("*.yaml")))]
            _output |= {load_yaml_config(x)["task"] for x in path}
    return list(_output)
