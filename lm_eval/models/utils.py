import collections
import fnmatch
import gc
import time
from functools import wraps
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
import transformers

from lm_eval.utils import eval_logger


def chunks(iter, n: int = 0, fn=None):
    """
    Divides an iterable into chunks of specified size or based on a given function.
    Useful for batching

    Parameters:
    - iter: The input iterable to be divided into chunks.
    - n: An integer representing the size of each chunk. Default is 0.
    - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

    Returns:
    An iterator that yields chunks of the input iterable.

    Example usage:
    ```
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for chunk in chunks(data, 3):
        print(chunk)
    ```
    Output:
    ```
    [1, 2, 3]
    [4, 5, 6]
    [7, 8, 9]
    [10]
    ```
    """
    arr = []
    for i, x in enumerate(iter):
        arr.append(x)
        if len(arr) == (fn(i, iter) if fn else n):
            yield arr
            arr = []

    if arr:
        yield arr


class MultiChoice:
    def __init__(self, choices) -> None:
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values) -> bool:
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                eval_logger.info("Available tasks to choose:")
                for choice in self.choices:
                    eval_logger.info(f"  - {choice}")
                raise ValueError("'{}' is not in task list".format(value))
        return True

    def __iter__(self) -> Iterator:
        for choice in self.choices:
            yield choice


class Grouper:
    """
    takes an array `arr` and function `fn` and returns a dictionary
    with keys fn(ob) for each ob in `arr` and with values `self.arr[key]` a list of all
    objects in `arr` satisfying `key == fn(ob)`.
    """

    def __init__(self, arr, fn) -> None:
        # self.orig_arr = arr
        self.size = len(arr)
        arr = list(enumerate(arr))

        def group_return_dict(arr, fn):
            res = collections.defaultdict(list)

            for ob in arr:
                res[fn(ob)].append(ob)
            return res

        arr = group_return_dict(arr, lambda x: fn(x[1]))

        # self.arr has format Dict[Tuple[int, <entry from orig. arr>]]
        self.arr = arr
        self._grouped = None

    def get_grouped(self):
        # return the contents but not indices for our grouped dict.
        if self._grouped:
            return self._grouped
        grouped = {}
        for key in self.arr.keys():
            # drop the index from each element of self.arr
            grouped[key] = [y[1] for y in self.arr[key]]
        self._grouped = grouped
        return grouped

    def get_original(self, grouped_dict):
        # take in a grouped dictionary with e.g. results for each key listed
        # in the same order as the instances in `self.arr`, and
        # return the results in the same (single list) order as `self.orig_arr`.
        res = [None] * self.size
        cov = [False] * self.size
        # orig = [None] * self.size

        assert grouped_dict.keys() == self.arr.keys()

        for key in grouped_dict.keys():
            for (ind, _), v in zip(self.arr[key], grouped_dict[key]):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def pad_and_concat(
    max_length: int,
    tensors: List[torch.Tensor],
    padding_side: Literal["right", "left"] = "right",
):
    """
    Method for padding a list of tensors given the maximum tensor
    length in the batch. Used for batching inputs and continuations in
    seq2seq models.
    """
    assert (
        padding_side == "left" or padding_side == "right"
    ), f"Unrecognized padding type: '{padding_side}' not 'left' or 'right'"

    for i, tensor in enumerate(tensors):
        if len(tensor.shape) == 2:
            tensor = tensor.squeeze(0)  # squeeze, in case passed [1, seq] size
        tensor_len = tensor.shape[0]
        if tensor_len < max_length:
            if padding_side == "right":
                # right-pad
                tensors[i] = torch.cat(
                    [
                        tensor,  # [seq]
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
            else:
                # left-pad
                tensors[i] = torch.cat(
                    [
                        torch.zeros(
                            max_length - tensor_len,
                            dtype=torch.long,
                            device=tensor.device,
                        ),  # [padding_length - seq]
                        tensor,  # [seq]
                    ],
                    dim=0,
                ).unsqueeze(0)
        else:
            tensors[i] = tensor.unsqueeze(0)

    return torch.cat(tensors, dim=0)


def clear_torch_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def divide(iterable, n) -> List[Iterator]:
    """Divide the elements from *iterable* into *n* parts, maintaining
    order.

        >>> group_1, group_2 = divide([1, 2, 3, 4, 5, 6], 2)
        >>> list(group_1)
        [1, 2, 3]
        >>> list(group_2)
        [4, 5, 6]

    If the length of *iterable* is not evenly divisible by *n*, then the
    length of the returned iterables will not be identical:

        >>> children = divide([1, 2, 3, 4, 5, 6, 7], 3)
        >>> [list(c) for c in children]
        [[1, 2, 3], [4, 5], [6, 7]]

    If the length of the iterable is smaller than n, then the last returned
    iterables will be empty:

        >>> children = divide([1, 2, 3], 5)
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]

    This function will exhaust the iterable before returning and may require
    significant storage. If order is not important, see :func:`distribute`,
    which does not first pull the iterable into memory.

    """
    if n < 1:
        raise ValueError("n must be at least 1")

    try:
        iterable[:0]
    except TypeError:
        seq = tuple(iterable)
    else:
        seq = iterable

    q, r = divmod(len(seq), n)

    ret = []
    stop = 0
    for i in range(1, n + 1):
        start = stop
        stop += q + 1 if i <= r else q
        ret.append(iter(seq[start:stop]))

    return ret


def retry_on_specific_exceptions(
    on_exceptions: List[Type[Exception]],
    max_retries: Optional[int] = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Optional[Callable[[Exception, float], Any]] = None,
):
    """Retry on an LLM Provider's rate limit error with exponential backoff
    For example, to use for OpenAI, do the following:
    ```
    from openai import RateLimitError

    # Recommend specifying max_retries to avoid infinite loops!
    @retry_on_specific_exceptions([RateLimitError], max_retries=3)
    def completion(...):
        # Wrap OpenAI completion function here
        ...
    ```
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = backoff_time
            attempt = 0
            while max_retries is None or attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except tuple(on_exceptions) as e:
                    if on_exception_callback is not None:
                        on_exception_callback(e, sleep_time)
                    time.sleep(sleep_time)
                    sleep_time *= backoff_multiplier
                    attempt += 1

        return wrapper

    return decorator


class Collator:
    """
    A class for reordering and batching elements of an array.

    This class allows for sorting an array based on a provided sorting function, grouping elements based on a grouping function, and generating batches from the sorted and grouped data.
    """

    def __init__(
        self,
        arr: List,
        sort_fn: Callable,
        group_fn: Callable = lambda x: x[1],
        grouping: bool = False,
    ) -> None:
        self.grouping = grouping
        self.fn = sort_fn
        self.group_fn = lambda x: group_fn(x[1])  # first index are enumerated indices
        self.reorder_indices: List = []
        self.size = len(arr)
        self.arr_with_indices: Iterable[Any] = tuple(enumerate(arr))  # [indices, (arr)]
        if self.grouping is True:
            self.group_by_index()

    def group_by_index(self) -> None:
        self.arr_with_indices = self.group(
            self.arr_with_indices, fn=self.group_fn, values=False
        )

    def get_batched(self, n: int = 1, batch_fn: Optional[Callable] = None) -> Iterator:
        """
        Generates and yields batches from the reordered array.

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn (Optional[Callable[[int, Iterable], int]]): A function to determine the size of each batch. Defaults to None.

        Yields:
        Iterator: An iterator over batches of reordered elements.
        """
        if self.grouping:
            for (
                key,
                values,
            ) in self.arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        else:
            values = self._reorder(self.arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def _reorder(self, arr: Union[List, Tuple[Tuple[int, Any], ...]]) -> List:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (Union[List, Tuple[Tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
        List: Yields reordered elements one by one.
        """
        arr = sorted(arr, key=lambda x: self.fn(x[1]))
        self.reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: List) -> List:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (List): The reordered array.

        Returns:
        List: The array with elements restored to their original order.
        """
        res = [None] * self.size
        cov = [False] * self.size

        for ind, v in zip(self.reorder_indices, newarr):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self.size

    @staticmethod
    def group(arr: Iterable, fn: Callable, values: bool = False) -> Iterable:
        """
        Groups elements of an iterable based on a provided function.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterable: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            try:
                hashable_dict = tuple(
                    (
                        key,
                        tuple(value)
                        if isinstance(value, collections.abc.Iterable)
                        else value,
                    )
                    for key, value in sorted(fn(ob).items())
                )
                res[hashable_dict].append(ob)
            except TypeError:
                res[fn(ob)].append(ob)
        if not values:
            return res
        return res.values()

    @staticmethod
    def get_chunks(_iter, n: int = 0, fn=None):
        """
        Divides an iterable into chunks of specified size or based on a given function.
        Useful for batching

        Parameters:
        - iter: The input iterable to be divided into chunks.
        - n: An integer representing the size of each chunk. Default is 0.
        - fn: A function that takes the current index and the iterable as arguments and returns the size of the chunk. Default is None.

        Returns:
        An iterator that yields chunks of the input iterable.

        Example usage:
        ```
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for chunk in chunks(data, 3):
            print(chunk)
        ```
        Output:
        ```
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
        ```
        """
        arr = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            arr.append(x)
            if len(arr) == (fn(i, _iter) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr
