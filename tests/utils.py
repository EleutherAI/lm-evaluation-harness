from typing import List
from lm_eval.utils import load_yaml_config
from pathlib import Path
from typing import Union
import os


# {{{CI}}}
# This is the path where the output for the changed files for the tasks folder is stored
# FILE_PATH = file_path = ".github/outputs/tasks_all_changed_and_modified_files.txt"


# reads a text file and returns a list of words
# used to read the output of the changed txt from tj-actions/changed-files
def load_changed_files(file_path: str) -> List[str]:
    with open(file_path, "r") as f:
        content = f.read()
        words_list = [x for x in content.split()]
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


def new_tasks() -> Union[List[str], None]:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        # If tasks folder has changed then we get the list of files from FILENAME
        # and parse the yaml files to get the task names.
        return parser(load_changed_files(FILENAME))
    elif os.getenv("API") is not None:
        # Or if API has changed then we set the ENV variable API to True
        # and run  given tasks.
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    # if both not true just do arc_easy
    else:
        return
