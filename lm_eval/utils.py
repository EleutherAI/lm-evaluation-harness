from __future__ import annotations

import collections
import fnmatch
import functools
import hashlib
import importlib.util
import inspect
import json
import logging
import os
import re
from collections.abc import Generator, Mapping
from dataclasses import asdict, is_dataclass
from functools import wraps
from itertools import islice
from typing import Any, Callable, TypeVar

import numpy as np
from jinja2 import BaseLoader, Environment, StrictUndefined


K = TypeVar("K")
V = TypeVar("V")


SPACING = " " * 47

HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def wrap_text(string: str, width: int = 140, **kwargs) -> str | None:
    """
    Wraps the given string to the specified width.
    """
    import textwrap

    return textwrap.fill(
        inspect.cleandoc(string),
        width=width,
        initial_indent="",
        subsequent_indent=" " * 8,
        break_long_words=False,
        break_on_hyphens=False,
        **kwargs,
    )


def get_logger(level: str | None = None) -> logging.Logger:
    """
    Get a logger with a stream handler that captures all lm_eval logs.

    Args:
        level (Optional[str]): The logging level.
    Example:
        >>> logger = get_logger("INFO")
        >>> logger.info("Log this")
        INFO:lm_eval:Log this!

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger("lm_eval")
    if not logger.hasHandlers():
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
    if level is not None:
        level = getattr(logging, level.upper())
        logger.setLevel(level)
    return logger


def setup_logging(verbosity=logging.INFO, suppress_third_party=True):
    """
    Configure logging for the lm_eval CLI application.

    WARNING: This function is intended for CLI use only. Library users should
    use get_logger() instead to avoid interfering with their application's
    logging configuration.

    Args:
        verbosity: Log level (int) or string name. Can be overridden by LOGLEVEL env var.
        suppress_third_party: Whether to suppress verbose third-party library logs.

    Returns:
        logging.Logger: The configured lm_eval logger instance.
    """
    # Validate verbosity parameter
    if isinstance(verbosity, str):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        verbosity = level_map.get(verbosity.upper(), logging.INFO)
    elif not isinstance(verbosity, int):
        verbosity = logging.INFO

    # Get log level from environment or use default
    if log_level_env := os.environ.get("LOGLEVEL", None):
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = level_map.get(log_level_env.upper(), verbosity)
    else:
        log_level = verbosity

    # Get the lm_eval logger directly
    logger = logging.getLogger("lm_eval")

    # Configure custom formatter
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.name = record.name.removeprefix("lm_eval.")
            return super().format(record)

    formatter = CustomFormatter(
        "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    # Check if handler already exists to prevent duplicates
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) for h in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        # For CLI use, we disable propagation to avoid duplicate messages
        logger.propagate = False

    # Set the logger level
    logger.setLevel(log_level)

    # Optionally suppress verbose third-party library logs
    if suppress_third_party and log_level == logging.DEBUG:
        third_party_loggers = ["urllib3", "filelock", "fsspec"]
        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    return logger


def hash_string(string: str) -> str:
    return hashlib.sha256(string.encode("utf-8")).hexdigest()


def escaped_split(text, sep_char, maxsplit=-1):
    """Split text into a list on occurrences of the given separation
    character `sep_char`. The separation character may be escaped by a
    backslash to avoid splitting at that location.

    The separation character must be a string of size 1.

    If `maxsplit` is given, at most `maxsplit` splits are done (thus,
    the list will have at most `maxsplit + 1` elements). If `maxsplit`
    is not specified or less than 0, then there is no limit on the
    number of splits (all possible splits are made).
    """
    assert len(sep_char) == 1, (
        "separation string must be a single character for escaped splitting"
    )

    if maxsplit == 0:
        return text
    maxsplit = max(0, maxsplit)

    return re.split(r"(?<!\\)" + sep_char, text, maxsplit=maxsplit)


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg


def handle_non_serializable(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def sanitize_list(sub):
    """
    Takes possible nested list and recursively converts all inner component to strings
    """
    if isinstance(sub, list):
        return [sanitize_list(item) for item in sub]
    if isinstance(sub, tuple):
        return tuple(sanitize_list(item) for item in sub)
    else:
        return str(sub)


def simple_parse_args_string(args_string: str | None) -> dict:
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    if args_string is None:
        return {}
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        kv[0]: handle_arg_string("=".join(kv[1:]))
        for kv in [arg.split("=") for arg in arg_list]
    }
    return args_dict


def join_iters(iters):
    for iter in iters:
        yield from iter


def group(arr, fn):
    res = collections.defaultdict(list)

    for ob in arr:
        res[fn(ob)].append(ob)

    return list(res.values())


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns: list[str], source_list: list[str]) -> list[str]:
    if isinstance(patterns, str):
        patterns = [patterns]

    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


def softmax(x) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def general_detokenize(string: str) -> str:
    string = string.replace(" n't", "n't")
    string = string.replace(" )", ")")
    string = string.replace("( ", "(")
    string = string.replace('" ', '"')
    string = string.replace(' "', '"')
    string = re.sub(r" (['.,])", r"\1", string)
    return string


def get_file_task_name(filename: str) -> str:
    """
    Given the sample results filenames, extracts and returns the task name.
    """
    return filename[filename.find("_") + 1 : filename.rfind("_")]


def get_file_datetime(filename: str) -> str:
    """
    Given the results and sample results filenames, extracts and returns the datetime.
    """
    return filename[filename.rfind("_") + 1 :].replace(".jsonl", "")


def sanitize_model_name(model_name: str) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    return re.sub(r"[\"<>:/|\\?*\[\]]+", "__", model_name)


def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)


def get_latest_filename(filenames: list[str]) -> str:
    """
    Given a list of filenames, returns the filename with the latest datetime.
    """
    return max(filenames, key=lambda f: get_file_datetime(f))


def get_results_filenames(filenames: list[str]) -> list[str]:
    """
    Extracts filenames that correspond to aggregated results.
    """
    return [f for f in filenames if "/results_" in f and ".json" in f]


def get_sample_results_filenames(filenames: list[str]) -> list[str]:
    """
    Extracts filenames that correspond to sample results.
    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def get_rolling_token_windows(
    token_list: list[int], prefix_token: int, max_seq_len: int, context_len: int
) -> Generator[tuple[list[int], list[int]], None, None]:
    """
    - context_len allows for a rolling window context, allowing each prediction window to potentially
      condition on some context

    :param token_list: list
        List of tokens to be PREDICTED
    :param max_seq_len: int
        max_seq_len of model (or max_seq_len we want to use)
    :param context_len: int
        Amount of desired token context for prediction. Needs to be at least 1.
    :param prefix_token: token
        Dummy token like <eos> so the first token has something to condition on
    :return: generator
        Generator of tuples
            (input_tokens, pred_tokens)
        Note: Score only the last len(pred_tokens) logits of the LM
    """
    assert 1 <= context_len <= max_seq_len
    if not token_list:
        return
    # +1 offset, going from input->preds
    pred_len = max_seq_len - context_len + 1
    predicted = 0

    # Special handling for first window: predict all tokens
    first_seq_len = min(max_seq_len, len(token_list))
    yield [prefix_token] + token_list[: first_seq_len - 1], token_list[:first_seq_len]
    predicted += first_seq_len

    while predicted < len(token_list):
        window_pred_len = min(len(token_list) - predicted, pred_len)
        window_end = predicted + window_pred_len

        yield (
            token_list[window_end - max_seq_len - 1 : window_end - 1],
            token_list[window_end - window_pred_len : window_end],
        )
        predicted += window_pred_len


def make_disjoint_window(
    pair: tuple[list[int], list[int]],
) -> tuple[list[int], list[int]]:
    """Takes output from get_rolling_token_windows and makes the context not overlap with the continuation"""
    a, b = pair
    return a[: len(a) - (len(b) - 1)], b


class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Provides a proper json encoding for the loggers and trackers json dumps.
    Notably manages the json encoding of dataclasses.
    """

    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)


class Reorderer:
    def __init__(self, arr: list[Any], fn: Callable) -> None:
        """Reorder an array according to some function

        Args:
            arr (List[Any]): The initial array
            fn (Callable[[Any], Any]): A function to determine the priority of elements
        """
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        # arr = [([y[0] for y in x], x[0][1]) for x in arr]
        # TODO: overhaul reorderer. It currently grouped requests by content but we don't want this
        arr = [([y[0]], x[0][1]) for x in arr for y in x]
        arr.sort(key=lambda x: fn(x[1]))

        self.arr = arr

    def get_reordered(self):
        """Gets the reordered array

        Returns:
            List[Any]: The reordered array
        """
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        """Restores the original order of a new array based on the old array's order

        Args:
            newarr (List[Any]): The array to be restored

        Returns:
            List[Any]: The array restored to the original order
        """
        res = [None] * self.size
        cov = [False] * self.size

        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True

        assert all(cov)

        return res


def make_table(result_dict, column: str = "results", sort_results: bool = False):
    """Generate table of results."""
    from pytablewriter import LatexTableWriter, MarkdownTableWriter

    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically by task or group name.
        # NOTE: we default here to false, because order matters for multi-level table printing a la mmlu.
        # sorting here would mess that up
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "    N/A")
        n = str(result_dict.get("n-shot", " ").get(k, " "))
        # TODO: fix this
        # higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            # hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")
            # TODO: fix
            hib = "↑"

            v = f"{v:.4f}" if isinstance(v, float) else v

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                se = "   N/A" if se == "N/A" else f"{se:.4f}"
                values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def positional_deprecated(fn):
    """
    A decorator to nudge users into passing only keyword args (`kwargs`) to the
    wrapped function, `fn`.
    """

    wraps(fn)

    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


def create_iterator(
    raw_iterator: collections.Iterator,
    *,
    rank: int = 0,
    world_size: int = 1,
    limit: int | None = None,
) -> islice:
    """
    Method for creating a (potentially) sliced and limited
    iterator from a raw document iterator. Used for splitting data
    among ranks in multigpu setting or only pulling a sample of documents
    """
    return islice(raw_iterator, rank, limit, world_size)


def weighted_f1_score(items):
    from sklearn.metrics import f1_score

    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds, average="weighted")
    return fscore


def convert_pil_to_hash(value):
    from io import BytesIO

    img_bytes = BytesIO()
    value.save(img_bytes, format="PNG")
    return hashlib.sha256(str(img_bytes).encode()).hexdigest()


def convert_bytes_to_hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def hash_dict_images(data_dict):
    """
    Create a deep copy of `data_dict` where all bytes and PIL.Image.Image values
    are replaced by their respective hashes using the provided converter functions.

    Parameters:
        data_dict (dict): The input dictionary with arbitrary nesting of dicts and lists.

    Returns:
        dict: A new dictionary with the same structure as `data_dict`, but with all
              bytes and PIL.Image.Image objects replaced by their hashes.
    """

    def _process_value(value):
        # Bytes -> hash
        from PIL import Image

        if isinstance(value, (bytes, bytearray)):
            return convert_bytes_to_hash(value)
        # PIL Image -> hash
        if isinstance(value, Image.Image):
            return convert_pil_to_hash(value)
        # Nested dictionary -> recurse
        if isinstance(value, dict):
            return {k: _process_value(v) for k, v in value.items()}
        # List or tuple -> recurse, preserving type
        if isinstance(value, list):
            return [_process_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_process_value(v) for v in value)
        # Other types remain unchanged
        return value

    # Ensure the top-level is a dict
    if not isinstance(data_dict, dict):
        raise TypeError("Input must be a dictionary")

    return (
        {key: _process_value(val) for key, val in data_dict.items()}
        if importlib.util.find_spec("PIL")
        else data_dict
    )


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


@functools.lru_cache(maxsize=256)
def _compile_tpl(src: str):
    return apply_template._env.from_string(src)


def apply_template(template: str, doc: dict) -> str:
    if not hasattr(apply_template, "_env"):
        apply_template._env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
        apply_template._env.filters["regex_replace"] = regex_replace

    return _compile_tpl(template).render(**doc)


def validate_index(index: int, length: int) -> int:
    return index if index < length else -100


@functools.lru_cache(maxsize=10)
def get_parameter_info(func: Callable, param_name: str) -> tuple[bool, Any | None]:
    """
    Check if a callable has a specific parameter and get its default value.

    Args:
        func: The callable to inspect
        param_name: Name of the parameter to look for

    Returns:
        Tuple of (has_parameter, default_value)
        - has_parameter: True if the parameter exists
        - default_value: The default value if it exists, None otherwise
    """
    import inspect

    if not callable(func):
        return False, None

    try:
        sig = inspect.signature(func)
        if param_name in sig.parameters:
            param = sig.parameters[param_name]
            if param.default != inspect.Parameter.empty:
                return True, param.default
            else:
                return True, None
        return False, None
    except (ValueError, TypeError):
        # Some built-in functions don't have accessible signatures
        return False, None


def get_parameter_default(func: Callable, param_name: str, fallback: Any = None) -> Any:
    """
    Get the default value of a parameter, with a fallback if not found.

    Args:
        func: The callable to inspect
        param_name: Name of the parameter
        fallback: Value to return if parameter doesn't exist or has no default

    Returns:
        The parameter's default value or the fallback
    """
    has_param, default = get_parameter_info(func, param_name)
    if has_param and default is not None:
        return default
    return fallback


def merge_dict_values(*args: Mapping[K, V]) -> dict[K, list[V]]:
    """
    Merges dictionaries recursively.
    """
    from collections import defaultdict

    out = defaultdict(list)
    for d in args:
        for k, v in d.items():
            out[k].append(v)
    return out
