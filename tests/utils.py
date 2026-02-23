import os

from lm_eval.tasks._yaml_loader import load_yaml


# {{{CI}}}
# This is the path where the output for the changed files for the tasks folder is stored
# FILE_PATH = file_path = ".github/outputs/tasks_all_changed_and_modified_files.txt"


# reads a text file and returns a list of words
# used to read the output of the changed txt from tj-actions/changed-files
def load_changed_files(file_path: str) -> list[str]:
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
        words_list = list(content.split())
    return words_list


# checks the txt file for list of changed files.
# if file ends with .yaml then check yaml and load the config.
# if the config task is a string, it's a task config.
# if the config task is a list, it's a group config.
def parser(full_path: list[str]) -> list[str]:
    _output = set()
    for x in full_path:
        if x.endswith(".yaml") and os.path.exists(x):
            config = load_yaml(x, resolve_func=False)
            if isinstance(config["task"], str):
                _output.add(config["task"])
            elif isinstance(config["task"], list):
                _output.add(config["group"])
    return list(_output)


def new_tasks() -> list[str] | None:
    FILENAME = ".github/outputs/tasks_all_changed_and_modified_files.txt"
    if os.path.exists(FILENAME):
        # If tasks folder has changed then we get the list of files from FILENAME
        # and parse the yaml files to get the task names.
        return parser(load_changed_files(FILENAME))
    if os.getenv("API") is not None:
        # Or if API has changed then we set the ENV variable API to True
        # and run  given tasks.
        return ["arc_easy", "hellaswag", "piqa", "wikitext"]
    # if both not true just do arc_easy
    return None
