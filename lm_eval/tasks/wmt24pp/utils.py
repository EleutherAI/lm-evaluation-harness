"""Utilities for the WMT24++ translation tasks.

This module provides a small helper used as `custom_dataset` in YAML-based
ConfigurableTasks. It loads the `google/wmt24pp` dataset for a specific
English→X language pair, filters bad sources, and returns a split dict
compatible with `ConfigurableTask`.
"""

from __future__ import annotations

from typing import Any, Dict

from datasets import Dataset, load_dataset


def load_wmt24pp_dataset(*, lang_pair: str, split: str = "train", **kwargs: Any) -> Dict[str, Dataset]:
    """Load and filter the WMT24++ dataset for a specific language pair.

    Parameters
    ----------
    lang_pair:
        Exact value of the `lp` field / HF config name, e.g. "en-de_DE".
    split:
        Dataset split name to load. WMT24++ exposes a single split ("train"),
        which we treat as the evaluation split.
    **kwargs:
        Extra keyword arguments forwarded to `load_dataset`.

    Returns
    -------
    dict[str, datasets.Dataset]
        A mapping from split name to filtered dataset, as expected by
        `ConfigurableTask.custom_dataset`.
    """
    # For WMT24++, the config name is the language pair (`lang_pair`).
    # Ignore extraneous kwargs coming from global metadata (e.g. model args
    # like `pretrained`, `dtype`, etc.). We only pass arguments that
    # `load_dataset` for this builder actually expects.
    _ = kwargs  # intentionally unused for now

    ds = load_dataset("google/wmt24pp", lang_pair, split=split)

    # Filter out bad sources as recommended by the dataset authors.
    ds = ds.filter(lambda ex: not ex["is_bad_source"])

    return {split: ds}
