import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from torch.utils.collect_env import get_pretty_env_info
from transformers import __version__ as trans_version


logger = logging.getLogger(__name__)


def remove_none_pattern(input_string: str) -> Tuple[str, bool]:
    """Remove the ',none' substring from the input_string if it exists at the end.

    Args:
        input_string (str): The input string from which to remove the ',none' substring.

    Returns:
        Tuple[str, bool]: A tuple containing the modified input_string with the ',none' substring removed
                          and a boolean indicating whether the modification was made (True) or not (False).
    """
    # Define the pattern to match ',none' at the end of the string
    pattern = re.compile(r",none$")

    # Use sub() to replace ',none' with an empty string
    result = re.sub(pattern, "", input_string)

    # check if the input_string changed
    removed = result != input_string

    return result, removed


def _handle_non_serializable(o: Any) -> Union[int, str, list]:
    """Handle non-serializable objects by converting them to serializable types.

    Args:
        o (Any): The object to be handled.

    Returns:
        Union[int, str, list]: The converted object. If the object is of type np.int64 or np.int32,
            it will be converted to int. If the object is of type set, it will be converted
            to a list. Otherwise, it will be converted to str.
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def get_commit_from_path(repo_path: Union[Path, str]) -> Optional[str]:
    try:
        git_folder = Path(repo_path, ".git")
        if git_folder.is_file():
            git_folder = Path(
                git_folder.parent,
                git_folder.read_text(encoding="utf-8").split("\n")[0].split(" ")[-1],
            )
        if Path(git_folder, "HEAD").exists():
            head_name = (
                Path(git_folder, "HEAD")
                .read_text(encoding="utf-8")
                .split("\n")[0]
                .split(" ")[-1]
            )
            head_ref = Path(git_folder, head_name)
            git_hash = head_ref.read_text(encoding="utf-8").replace("\n", "")
        else:
            git_hash = None
    except Exception as err:
        logger.debug(
            f"Failed to retrieve a Git commit hash from path: {str(repo_path)}. Error: {err}"
        )
        return None
    return git_hash


def get_git_commit_hash():
    """
    Gets the git commit hash of your current repo (if it exists).
    Source: https://github.com/EleutherAI/gpt-neox/blob/b608043be541602170bfcfb8ec9bf85e8a0799e0/megatron/neox_arguments/neox_args.py#L42
    """
    try:
        git_hash = subprocess.check_output(["git", "describe", "--always"]).strip()
        git_hash = git_hash.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # FileNotFoundError occurs when git not installed on system
        git_hash = get_commit_from_path(os.getcwd())  # git hash of repo if exists
    return git_hash


def add_env_info(storage: Dict[str, Any]):
    try:
        pretty_env_info = get_pretty_env_info()
    except Exception as err:
        pretty_env_info = str(err)
    transformers_version = trans_version
    upper_dir_commit = get_commit_from_path(
        Path(os.getcwd(), "..")
    )  # git hash of upper repo if exists
    added_info = {
        "pretty_env_info": pretty_env_info,
        "transformers_version": transformers_version,
        "upper_git_hash": upper_dir_commit,  # in case this repo is submodule
    }
    storage.update(added_info)
