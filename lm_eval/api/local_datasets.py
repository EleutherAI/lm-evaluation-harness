import datasets
import hashlib
import json
import os
import yaml


from typing import Optional


_DEFAULT_BASE_PATH = os.path.expanduser("~/.lm_eval/datasets")


def save_dataset_to_disk(
    dataset,
    local_base_dir=None,
    hf_path=None,
    hf_name=None,
    hf_dataset_kwargs=None):
    """
    Save a dataset to disk in a subdirectory defined by a hash of its parameters.

    Parameters:
        dataset: The dataset object to be saved.
        local_base_dir (str or None): The base directory in which to save datasets.
        hf_path (str or None): The dataset path used during creation.
        hf_name (str or None): The dataset name used during creation.
        hf_dataset_kwargs (dict or None): Additional keyword arguments used during creation.

    Returns:
        str: The full path to the directory where the dataset was saved.
    """
    # Create a hash for the dataset using the given parameters.
    dataset_hash = _create_dataset_hash(hf_path, hf_name, hf_dataset_kwargs)
    local_base_dir = _get_local_base_dir(local_base_dir)
    dataset_dir = os.path.join(local_base_dir, dataset_hash)

    # Create the directory if it doesn't already exist.
    os.makedirs(dataset_dir, exist_ok=True)

    # Save the dataset to the specified directory.
    dataset.save_to_disk(dataset_dir)

    # Save configuration information to the directory.
    with open(os.path.join(dataset_dir, "config.json"), "w") as f:
        json.dump({
            "hf_path": hf_path,
            "hf_name": hf_name,
            "hf_dataset_kwargs": hf_dataset_kwargs
        }, f, indent=4)

    return dataset_dir


def load_dataset_from_disk(
    local_base_dir=None, hf_path=None, hf_name=None, hf_dataset_kwargs=None
):
    """
    Load a dataset from disk based on the hash of its creation parameters.

    Parameters:
        local_base_dir (str): The base directory where the dataset is saved.
        hf_path (str or None): The dataset path used during creation.
        hf_name (str or None): The dataset name used during creation.
        hf_dataset_kwargs (dict or None): Additional keyword arguments used during creation.

    Returns:
        The loaded dataset.

    Raises:
        FileNotFoundError: If the dataset directory doesn't exist.
    """
    # Recreate the hash for the dataset.
    dataset_hash = _create_dataset_hash(hf_path, hf_name, hf_dataset_kwargs)
    local_base_dir = _get_local_base_dir(local_base_dir)
    dataset_dir = os.path.join(local_base_dir, dataset_hash)

    # Check if the directory exists.
    if not os.path.exists(dataset_dir):
        fmt_kwargs = None
        if hf_dataset_kwargs is not None:
            fmt_kwargs = ",".join([f"{k}={v}" for k, v in hf_dataset_kwargs.items()])
        missing = "/".join(x for x in [hf_path, hf_name, fmt_kwargs] if x is not None)
        raise FileNotFoundError(f"No dataset found for {missing}.")

    # Load the dataset from disk.
    dataset = datasets.load_from_disk(dataset_dir)
    return dataset


def _create_dataset_hash(path, name, dataset_kwargs):
    """
    Create a hash from the dataset creation parameters.

    Parameters:
        path (str or None): The dataset path.
        name (str or None): The dataset name.
        dataset_kwargs (dict or None): Additional keyword arguments used to load the dataset.

    Returns:
        str: A hexadecimal hash string uniquely identifying the dataset configuration.
    """
    # Ensure dataset_kwargs is a dict (use an empty dict if None)
    dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}

    # Create a stable string representation by dumping a dict with sorted keys.
    cfg = {"path": path, "name": name, "dataset_kwargs": dataset_kwargs}
    hash_input = yaml.dump(cfg, Dumper=yaml.Dumper)

    # Create a hash of the input string.
    hash_value = _md5(hash_input)
    return hash_value


def _md5(to_hash: str, encoding: str = "utf-8") -> str:
    """Copied from axolotl"""
    try:
        return hashlib.md5(to_hash.encode(encoding), usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.md5(to_hash.encode(encoding)).hexdigest()  # nosec


def _get_local_base_dir(local_base_dir: Optional[str] = None) -> str:
    """
    Get the base path for saving datasets.

    Parameters:
        local_base_dir (str or None): The base directory in which to create the dataset directory.
            If None, the default base path is used.

    Returns:
        str: The base path for saving datasets.
    """
    if local_base_dir is not None:
        return local_base_dir
    return os.getenv("LM_EVAL_DATASETS_PATH", _DEFAULT_BASE_PATH)
