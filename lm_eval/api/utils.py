import collections
import pathlib
import re
import sys
import torch
from typing import Callable, Final, Iterable, List, Optional, Tuple, Union
from collections.abc import MutableMapping
from transformers import set_seed as transformers_set_seed


# General Utils


class ExitCodeError(Exception):
    pass


# Reproducibility utils


DEFAULT_SEED: Final[int] = 1234


def set_seed(seed: Optional[int] = DEFAULT_SEED):
    transformers_set_seed(seed)


# Token Utils


def general_detokenize(s: str) -> str:
    s = s.replace(" n't", "n't")
    s = s.replace(" )", ")")
    s = s.replace("( ", "(")
    s = s.replace('" ', '"')
    s = s.replace(' "', '"')
    s = re.sub(r" (['.,])", r"\1", s)
    return s


def get_rolling_token_windows(
    token_list: List[int], prefix_token: int, max_seq_len: int, context_len: int
) -> Iterable[Tuple[List[int], List[int]]]:
    """Returns a generator of rolling windows of length `max_seq_len` from a list of tokens.

    Args:
        token_list (List[int]):
            List of tokens to be predicted.
        prefix_token (int):
            Dummy token like <eos> so the first token has something to condition
            on.
        max_seq_len (int):
            The maximum sequence length of the model or a length we want to use.
        context_len (int):
            Amount of desired token context for prediction. Needs to be at least 1.
            This allows for a rolling window context, letting each prediction
            window to potentially condition on some context.

    Returns:
        Generator of tuples: (input_tokens, pred_tokens)
        NOTE: Score only the last len(pred_tokens) logits of the LM.
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


def split_and_pad_windows(
    windows: List[Tuple[str, str]], pad_token_id: int, max_seq_len: int
) -> Tuple[List[int], List[int]]:
    """Splits and pads a sequence of rolling context and continuation windows
    from `get_rolling_token_windows`.

    Example:
        [
            ([1] , [23, 19, 3]),  # (context, continuation)
            ([43], [2, 4]])
        ]

    Output:
        [
            [[1],[43]],                # Split & padded contexts.
            [[23, 19, 3], [2, 4, 1]]`  # Split & padded continuations.
        ]
        where `1` = `pad_token` id.

    Args:
        windows (List[Tuple[str, str]]):
            A generator of rolling `(context, continuation)` token windows
            (tuples).
        pad_token_id (int):
            The token id to pad with.
        max_seq_len (int):
            The maximum sequence length of the model or a length we want to use.

    Returns:
        A tuple of (context, continuation) padding windows.
    """
    contexts, continuations = zip(*windows)
    contexts, continuations = list(contexts), list(continuations)

    # Pad contexts:
    rollover_context = contexts[-1]
    rollover_context_size = len(rollover_context)
    # Handle empty final context token list - just add 1 token.
    if rollover_context_size == 0:
        contexts[-1] += [pad_token_id]
    elif rollover_context_size > 1:
        for i in range(len(contexts[:-1])):
            contexts[i] += [pad_token_id] * (rollover_context_size - len(contexts[i]))

    # Pad continuations:
    rollover_continuation = continuations[-1]
    rollover_continuation_size = len(rollover_continuation)
    is_multiple_windows = len(continuations) > 1
    if rollover_continuation_size < max_seq_len and is_multiple_windows:
        continuations[-1] = rollover_continuation + [pad_token_id] * (
            max_seq_len - rollover_continuation_size
        )
    return contexts, continuations


def make_disjoint_window(pair):
    """Takes output from get_rolling_token_windows and makes the context not
    overlap with the continuation.
    """
    a, b = pair
    return a[: -(len(b) - 1)], b


def select_continuation_from_batch_left_padding(
    generations: Union[List[List[int]], torch.Tensor], max_context_size: int
):
    """Select the continuation from the batch, removing prompts of different lengths.

    Args:
        generations (Union[List[List[int]], torch.Tensor]):
            A tensor or list-of-lists of shape [batch_size, sequence length].
        max_context_size (int):
            The size of the biggest context; generations will proceed from that
            index.

    Example:
        PAD     PAD Continue : The dog chased the cat  [every       day of the week]
        Riddle  me    this   : The  dog chased the  cat [yesterday] PAD PAD PAD PAD

    Output:
        [every day of the week]
        [yesterday]  PAD PAD PAD PAD
    """
    return generations[:, max_context_size:]


# Container Utils


class Reorderer:
    def __init__(self, arr, fn):
        self.size = len(arr)
        arr = list(enumerate(arr))
        arr = group(arr, lambda x: fn(x[1]))
        arr = [([y[0] for y in x], x[0][1]) for x in arr]
        arr.sort(key=lambda x: fn(x[1]))
        self.arr = arr

    def get_reordered(self):
        return [x[1] for x in self.arr]

    def get_original(self, newarr):
        res = [None] * self.size
        cov = [False] * self.size
        for (inds, _), v in zip(self.arr, newarr):
            for ind in inds:
                res[ind] = v
                cov[ind] = True
        assert all(cov)
        return res


def flatten(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "_",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def join_iters(iterables: Iterable) -> List:
    for iterable in iterables:
        yield from iterable


def chunks(iterable: Iterable, n: int) -> List:
    arr = []
    for x in iterable:
        arr.append(x)
        if len(arr) == n:
            yield arr
            arr = []
    if arr:
        yield arr


def group(arr: Iterable, fn: Callable) -> List:
    res = collections.defaultdict(list)
    for ob in arr:
        res[fn(ob)].append(ob)
    return list(res.values())


# CLI utils


def cli_template_names(
    task_name: str, template_names: str, template_idx: int = None
) -> List[str]:
    """Returns a selection of template names for a given task and comma-
    separated string of template names.

    Example:
        cli_template_names("task", "A,B,C") -> ["A", "B", "C"]

    Args:
        task_name (str):
            Name of the task from which to retrieve template names.
        template_names (str):
            A string of template names separated by a comma if multiple names
            are given.
            General Selectors:
                "all_templates":
                    Returns all templates for the task.
                "original_templates":
                    Returns all templates with formatting that matches the
                    original task design.
        template_idx (int, optional, defaults to None):
            If given, returns only the template at the given index.

    Returns:
        A list of template names.
    """
    import lm_eval.tasks

    if template_names == "all_templates":
        selections = lm_eval.tasks.list_templates(task_name)
    elif template_names == "original_templates":
        templates = lm_eval.tasks.get_templates(task_name)
        selections = []
        for name in templates.all_template_names:
            if templates[name].metadata.original_task is True:
                selections.append(name)
        if not selections:
            raise ValueError(f"No original task templates found for {task_name}")
    else:
        selections = template_names.split(",")
    if template_idx is not None:
        selections = [selections[template_idx]]
    return selections


def parse_cli_args_string(args: str) -> dict:
    """Parses a string in the following format to a kwargs dictionary.
    "args1=val1,arg2=val2"
    """
    # Remove leading whitespace but not trailing in case a `val` contains necessary whitespace.
    args = args.lstrip()
    if not args:
        return {}
    arg_list = args.split(",")
    args_dict = {}
    for arg in arg_list:
        # Split on the first `=` to allow for `=`s in `val`.
        k, v = arg.split("=", 1)
        args_dict[k] = str_to_builtin_type(v)
    return args_dict


def str_to_builtin_type(s: str) -> str:
    for fn in (to_bool, int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    return s


# https://stackoverflow.com/questions/7019283/automatically-type-cast-parameters-in-python
def to_bool(s: str):
    if s == "True" or s == "true":
        return True
    if s == "False" or s == "false":
        return False
    raise ValueError(f"The input `{s}` is not of boolean form.")


# Test utils


def find_test_root(*, start_path: pathlib.Path) -> pathlib.Path:
    """Search upward in the directory tree to a maximum of three layers
    to find and return the package root (containing the 'tests' folder)
    """
    cur_path = start_path.resolve()
    max_layers = 3
    for _ in range(max_layers):
        if (cur_path / "tests" / "test_version_stable.py").exists():
            return cur_path
        else:
            cur_path = cur_path.parent.resolve()
    raise FileNotFoundError(
        f"Unable to find package root within {max_layers} upwards" + f"of {start_path}"
    )


def run_task_tests(*, task_list: List[str]):
    """Find the package root and run the tests for the given tasks."""
    import pytest

    package_root = find_test_root(start_path=pathlib.Path(__file__))
    task_string = " or ".join(task_list)
    args = [
        f"{package_root}/tests/test_version_stable.py",
        f"--rootdir={package_root}",
        "-k",
        f"{task_string}",
    ]
    sys.path.append(str(package_root))
    pytest_return_val = pytest.main(args)
    if pytest_return_val:
        raise ValueError(
            f"Not all tests for the specified tasks ({task_list}) ran successfully! Error code: {pytest_return_val}"
        )
