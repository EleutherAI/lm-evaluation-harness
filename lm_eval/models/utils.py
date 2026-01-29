from __future__ import annotations

import collections
import fnmatch
import itertools
import logging
import time
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
)

from typing_extensions import TypedDict

from lm_eval.utils import maybe_warn, warning_once


eval_logger = logging.getLogger(__name__)
T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    import torch
    from PIL import Image
    from transformers import PreTrainedTokenizerBase
    from transformers.configuration_utils import PretrainedConfig


class GenKwargs(TypedDict, total=False):
    do_sample: bool
    temperature: float
    # other alias' will be converted to `max_gen_toks`.
    max_gen_toks: int
    until: list[str]
    __extra_items__: Any


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
                raise ValueError(f"'{value}' is not in task list")
        return True

    def __iter__(self) -> Iterator:
        yield from self.choices


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
            for (ind, _), v in zip(self.arr[key], grouped_dict[key], strict=True):
                res[ind] = v
                cov[ind] = True
                # orig[ind] = _

        assert all(cov)
        # assert orig == self.orig_arr

        return res


def undistribute(iterable):
    """
    Undoes https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distribute .

    Re-interleaves results that have been split using more_itertools.distribute:
        >>> group_1, group_2 = distribute(2, [1, 2, 3, 4, 5, 6])
        >>> list(group_1)
        [1, 3, 5]
        >>> list(group_2)
        [2, 4, 6]
        >>> undistribute([group_1, group_2])
        [1, 2, 3, 4, 5, 6]

    Handles non-uniform component lengths:

        >>> children = distribute(3, [1, 2, 3, 4, 5, 6, 7])
        >>> [list(c) for c in children]
        [[1, 4, 7], [2, 5], [3, 6]]
        >>> undistribute(children)
        [1, 2, 3, 4, 5, 6, 7]

    Also handles when some iterables are empty:

        >>> children = distribute(5, [1, 2, 3])
        >>> [list(c) for c in children]
        [[1], [2], [3], [], []]
        >>> undistribute(children)
        [1, 2, 3]

    """

    return [
        x
        for x in itertools.chain.from_iterable(
            itertools.zip_longest(*[list(x) for x in iterable])
        )
        if x is not None
    ]


def retry_on_specific_exceptions(
    on_exceptions: list[type[Exception]],
    max_retries: int | None = None,
    backoff_time: float = 3.0,
    backoff_multiplier: float = 1.5,
    on_exception_callback: Callable[[Exception, float], Any] | None = None,
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

    Objects of this class have the group_by attribute which determines the method for grouping
    the data while batching it. Three options include "gen_kwargs", "contexts", or None:
        If group_by == "gen_kwargs" then requests will be grouped by gen_kwargs
        If group_by == "contexts" then requests will be grouped by context + cont[:-1]
        If None then requests will just be reordered by length descending.
    """

    def __init__(
        self,
        arr: Sequence[T],
        sort_fn: Callable[[T], Any] = lambda x: x,
        group_fn: Callable[[T], Any] = lambda x: x[1],
        group_by: Literal["gen_kwargs", "contexts"] | None = None,
    ) -> None:
        self._group_by = group_by
        # 0 indices are enumerated indices. Apply functions to original arr.
        self._sort_fn = lambda x: sort_fn(x[1])
        self._group_fn = lambda x: group_fn(x[1])
        self._reorder_indices: list[int] = []
        self._size = len(arr)
        self._arr_with_indices: dict | tuple[tuple[int, Any], ...] = tuple(
            enumerate(arr)
        )  # [indices, (arr)]
        if self._group_by == "contexts":
            self._group_by_context()
        elif self._group_by == "gen_kwargs":
            self._group_by_index()

    def _group_by_index(self) -> None:
        """Group the elements of a list based on their indices."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="gen_kwargs"
        )

    def _group_by_context(self) -> None:
        """Group the array with indices by context."""
        self._arr_with_indices = self.group(
            self._arr_with_indices, fn=self._group_fn, group_by="contexts"
        )

    def get_batched(
        self, n: int = 1, batch_fn: Callable[[int, Iterable[T]], int] | None = None
    ) -> Iterator[T]:
        """
        Generates and yields batches from the reordered array. The method of grouping and batching
        depends on the parameter `group_by`.
        If `group_by` is set to "gen_kwargs", it will batch the
        re-ordered values with same gen_kwargs for each batch.
        If `group_by` is "contexts", it caches the requests by context before batching.
        If `group_by` is neither "gen_kwargs" nor "contexts", it yields the reordered array

        Parameters:
        - n (int): The size of each batch. Defaults to 1.
        - batch_fn ([Callable[[int, Iterable], int]] | None): A function to determine the size of
          each batch. Defaults to None.

        Returns:
        Iterator: An iterator over batches of reordered elements grouped as per the `group_by`
                  attribute.

        Yields:
        List of batched elements according to the `group_by` attribute.
        """
        if self._group_by == "gen_kwargs":
            for (
                _,
                values,
            ) in self._arr_with_indices.items():  # type: ignore
                values = self._reorder(values)
                batch = self.get_chunks(values, n=n, fn=batch_fn)
                yield from batch
        elif self._group_by == "contexts":
            # Get one sample from each key.
            # Select longest continuation per group to ensure sufficient context logits
            values = self._reorder(
                [
                    max(value, key=lambda x: len(x[1][-1]))
                    for value in self._arr_with_indices.values()
                ]
            )
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch
        else:
            values = self._reorder(self._arr_with_indices)  # type: ignore
            batch = self.get_chunks(values, n=n, fn=batch_fn)
            yield from batch

    def get_cache(
        self,
        req_str: tuple[str, str],
        cxt_toks: list[int],
        cont_toks: list[int],
        logits: torch.Tensor,
    ) -> Iterator[tuple[tuple[str, str], list[int], torch.Tensor]]:
        """
        Retrieves cached single-token continuations and their associated arguments, updating indices as necessary.

        The behavior of this function varies depending on how the `group_by` attribute is set:

        - When `group_by` is "contexts":
            The function identifies single-token continuations by checking for keys that equate to
            [context+continuation][-1] and logs the indices for re-ordering.
            In this mode, this function can work in two scenarios:

            1. Cache Hit - Single Match:
                If a single matching context-continuation pair is found in the cache,
                the function yields the original arguments.

            2. Cache Hit - Multiple Matches:
                If multiple matching context-continuation pairs are found in the cache,
                the function expands the logits batch dimension to match the number of cache hits.
                It updates the original requests and continuation tokens.

        - When `group_by` is not set to "contexts":
            This method yields the original arguments, logits and continuation tokens,
            without checking for one-token continuations.

        Parameters:
        - req_str (tuple[str, str]): Original strings used for CachingLM.
        - cxt_toks (list[int]): Full context tokens used for lookup.
        - cont_toks (list[int]): Continuation tokens for which logits were generated.
        - logits (torch.Tensor [1, seq_length, vocab_size]): Logits generated by the model given context and continuation keys.

        Yields:
        - Iterator:
            - req_str (tuple[str, str]): strings used for CachingLM.
            - cont_toks (list[int]) : continuation tokens.
            - logits (torch.Tensor [1, seq_length, vocab_size]): The original logits (repeated cache hit times)
        """
        if self._group_by == "contexts":
            cache_hit: list[
                tuple[int, tuple[tuple[str, str], list[int], list[int]]]
            ] = self._arr_with_indices.pop(tuple(cxt_toks + cont_toks[:-1]))
            if (cache_size := len(cache_hit)) == 1:
                self._reorder_indices.extend(x[0] for x in cache_hit)
                yield req_str, cont_toks, logits
            else:
                # If we have matching requests then expand the batch dimension (no-op) and
                # yield each along with its corresponding args.
                multilogits = logits.expand(cache_size, -1, -1).chunk(cache_size)
                indices, req_str, cont_toks = zip(
                    *[(x[0], x[1][0], x[-1][-1]) for x in cache_hit], strict=True
                )
                self._reorder_indices.extend(indices)
                yield from zip(req_str, cont_toks, multilogits, strict=True)
        else:
            yield req_str, cont_toks, logits

    def _reorder(self, arr: list | tuple[tuple[int, Any], ...]) -> Iterator:
        """
        Reorders the elements in the array based on the sorting function.

        Parameters:
        - arr (list | tuple[tuple[int, Any], ...]]): The array or iterable to be reordered.

        Yields:
            Iterator
        """
        arr = sorted(arr, key=self._sort_fn)
        if self._group_by != "contexts":
            # If grouped by contexts then indices will be set in get_cache()
            self._reorder_indices.extend([x[0] for x in arr])
        yield from [x[1] for x in arr]

    def get_original(self, newarr: list) -> list:
        """
        Restores the original order of elements from the reordered list.

        Parameters:
        - newarr (list): The reordered array.

        Returns:
        list: The array with elements restored to their original order.
        """
        res = [None] * self._size
        cov = [False] * self._size

        for ind, v in zip(self._reorder_indices, newarr, strict=True):
            res[ind] = v
            cov[ind] = True

        assert all(cov)

        return res

    def __len__(self):
        return self._size

    @staticmethod
    def group(
        arr: Iterable[T],
        fn: Callable[[T], Sequence[T] | dict],
        group_by: Literal["gen_kwargs", "contexts"] = "gen_kwargs",
    ) -> dict:
        """
        Groups elements of an iterable based on a provided function.


        The `group_by` parameter determines the method of grouping.
        If `group_by` is "contexts", the elements are grouped by [context + cont][:-1].
        If `group_by` is "gen_kwargs", the elements are grouped based on the gen_kwargs dict.

        Parameters:
        - arr (Iterable): The iterable to be grouped.
        - fn (Callable): The function to determine the grouping.
        - values (bool): If True, returns the values of the group. Defaults to False.

        Returns:
        Iterator: An iterable of grouped elements.
        """
        res = collections.defaultdict(list)
        for ob in arr:
            # where ob == [context + cont]
            if group_by == "contexts":
                res[tuple(fn(ob))].append(ob)
            else:
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
                except (TypeError, AttributeError):
                    res[tuple(fn(ob))].append(ob)
        return res

    @staticmethod
    def get_chunks(
        _iter, n: int = 0, fn: Callable[[int, Iterable[T]], int] | None = None
    ) -> Iterator[T]:
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


def configure_pad_token(
    tokenizer: PreTrainedTokenizerBase,
    model_config: PretrainedConfig | None = None,
) -> PreTrainedTokenizerBase:
    """
    This function checks if the (Hugging Face) tokenizer has a padding token and sets it if not present.
    Some tokenizers require special handling.

    Args:
        tokenizer: The tokenizer for which the padding token is to be handled.
        model_config: The configuration of the model. Default is None.

    Returns:
        The tokenizer after the padding token has been handled.

    Raises:
        AssertionError: If the tokenizer is of type RWKVWorldTokenizer or Rwkv5Tokenizer and the padding token id is not 0.
    """
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # handle special cases
        if model_config and getattr(model_config, "model_type", None) == "qwen":
            # Qwen's trust_remote_code tokenizer does not allow for adding special tokens
            tokenizer.pad_token = "<|endoftext|>"
        elif (
            tokenizer.__class__.__name__ == "RWKVWorldTokenizer"
            or tokenizer.__class__.__name__ == "Rwkv5Tokenizer"
        ):
            # The RWKV world tokenizer, does not allow for adding special tokens / setting the pad token (which is set as 0)
            # The additional tokenizer name check is needed, as there exists rwkv4 models with neox tokenizer
            # ---
            # Note that the world tokenizer class name, might change in the future for the final huggingface merge
            # https://github.com/huggingface/transformers/pull/26963
            assert tokenizer.pad_token_id == 0
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    return tokenizer


def replace_placeholders(
    string: str, default_placeholder: str, image_token: str, max_images: int
):
    """
    A utility function used for local multimodal models. It locates all `placeholder` string
    occurrences in the given input `string_` and replaces the first `max_count` instances with
    `replacement`, and all subsequent occurrences with the empty string.

    This is used to replace <image> placeholder tags by model-specific image tokens like <|image_pad|>
    and to allow for only the first `max_count` images to be passed to a model if desired.

    :param string: The original string containing placeholders.
    :param default_placeholder: The placeholder text to be replaced.
    :param image_token: The token to replace the placeholder with.
    :param max_images: The maximum number of replacements to make.
    :return: The string with placeholders replaced.
    """
    count = 0
    result = []

    parts = string.split(default_placeholder)
    for part in parts[:-1]:  # Iterate through all but the last part
        result.append(part)
        if count < max_images:
            result.append(image_token)
            count += 1
        elif default_placeholder != image_token:
            result.append(default_placeholder)

    # Add the last part of the string
    result.append(parts[-1])
    return "".join(result)


def flatten_image_list(images: list[list]):
    """
    Takes in a list of lists of images, and returns a single list of all images in order.
    Used for some multimodal models like Llava-1.5 which expects this flattened-list format for its image processor.

    :param images: A list of lists of PIL images.
    :return: a list of PIL images, via concatenating all the sub-lists in order.
    """
    return [image for image_list in images for image in image_list]


def handle_stop_sequences(until: str | list[str] | None, eos: str | None) -> list[str]:
    """Ensures that the `until` parameter is a list of stop sequences and includes the EOS token."""
    if isinstance(until, str):
        until = [until]
    elif until is None:
        until = []
    elif not isinstance(until, list):
        raise ValueError(
            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
        )

    if eos is not None and eos not in until:
        until.append(eos)
    return until


def normalize_gen_kwargs(
    gen_kwargs: dict,
    default_max_gen_toks: int = 256,
) -> GenKwargs:
    """Normalize generation kwargs for consistent handling across model backends.

    Model implementations may have different expectations for generation parameters.

    Args:
        gen_kwargs: Raw generation kwargs from the request. Expected keys include:
            - do_sample: Whether to use sampling (vs greedy decoding) - Required
            - until (str | list[str]): Stop sequence(s) for generation.
            - max_gen_toks | max_new_tokens | max_tokens | max_completion_tokens: Maximum tokens to generate
            - temperature: Sampling temperature
            - Other backend-specific kwargs
        default_max_gen_toks: Default max_gen_toks if not specified in gen_kwargs.

    Returns:
        A normalized dict containing:
        - do_sample (bool): Whether to use sampling (bool)
        - until: list[str]: List of stop sequences.
        - max_gen_toks (int): Maximum tokens to generate (int)
        - temperature (float): Sampling temperature (float). Note: will always be set to 0.0 if do_sample=False or do_sample is not specified.
        - All other kwargs passed through unchanged

    Notes:
        - Accepts `max_gen_toks` and other aliases. Priority:
          max_gen_toks > max_new_tokens > max_tokens > max_completion_tokens.
          Output always uses `max_gen_toks`.
        - When `do_sample=False`, temperature is set to 0.0 for greedy decoding.
        - When temperature is 0.0 and `do_sample` is not specified, `do_sample` is set
          to False.
        - Model backends may further modify the returned dict as needed (e.g., vLLM
          removes `do_sample` since it uses temperature directly).
    """

    import copy

    kwargs = copy.deepcopy(gen_kwargs)

    until = kwargs.get("until", [])
    if not isinstance(until, list):
        until = [until]

    # Extract max_gen_toks from various aliases (priority order: max_gen_toks > max_new_tokens > max_tokens > max_completion_tokens)
    max_token_aliases = {
        "max_gen_toks": kwargs.pop("max_gen_toks", None),
        "max_new_tokens": kwargs.pop("max_new_tokens", None),  # used in HF
        "max_tokens": kwargs.pop(
            "max_tokens", None
        ),  # used by vllm, OpenAI API's and others
        "max_completion_tokens": kwargs.pop(
            "max_completion_tokens", None
        ),  # newer OpenAI API alias
        # note: `max_length` is also used by HF but has different semantics (prompt + generation)
    }
    provided = {k: v for k, v in max_token_aliases.items() if v is not None}

    if len(provided) > 1:
        warning_once(
            eval_logger,
            f"Multiple max token args provided: {provided}. Using first by priority (max_gen_toks > max_new_tokens > max_tokens > max_completion_tokens).",
        )

    max_gen_toks = int(next(iter(provided.values()), default_max_gen_toks))

    # Handle do_sample and temperature consistently
    do_sample: bool | None = kwargs.get("do_sample")
    temperature: float | None = float(kwargs.get("temperature", 0.0))

    match do_sample:
        case None:
            kwargs["do_sample"] = True if temperature > 0.0 else False  # noqa: SIM210
        # do_sample=False -> temperature=0.0
        case False:
            if temperature and temperature != 0.0:
                warning_once(
                    eval_logger,
                    f"{do_sample=}` but {temperature=}; setting `temperature` to 0.0 for greedy decoding. For non-greedy decoding, set `do_sample=True`.",
                )
            kwargs["temperature"] = 0.0
        case True:
            # do_sample=True -> use provided kwargs
            if temperature == 0.0:
                warning_once(
                    eval_logger,
                    f"{do_sample=}` but {temperature=}. For non-greedy sampling, set temperature > 0.0",
                )

    # Set normalized values
    kwargs["until"] = until
    kwargs["max_gen_toks"] = max_gen_toks

    return GenKwargs(**kwargs)  # type:ignore[missing-typed-dict-key]


def resize_image(
    image: Image.Image,
    width: int | None = None,
    height: int | None = None,
    max_dimension: int | None = None,
    keep_aspect_ratio: bool = True,
    resample_filter: int | None = None,
    min_width: int = 1,
    min_height: int = 1,
) -> Image.Image:
    """
    Resizes a PIL Image object with flexible options.

    Args:
        image: The PIL Image object to resize.
        width: Target width in pixels.
        height: Target height in pixels.
        max_dimension: Maximum size for the longer dimension of the image.
        keep_aspect_ratio: If True (default) and both width and height are provided,
                          the image is resized to fit within these dimensions while
                          maintaining its aspect ratio. If False, the image is stretched
                          to the exact width and height.
        resample_filter: The resampling filter to use for resizing.
                        Defaults to Image.BICUBIC.
        min_width: Minimum width for the resized image. Defaults to 1.
        min_height: Minimum height for the resized image. Defaults to 1.

    Returns:
        The resized PIL Image object. If no resize parameters are provided
        or if the image already meets the criteria, the original image is returned.

    Order of precedence for resizing:
    1. If width AND height are provided:
       - If keep_aspect_ratio is True: Fits image within bounds, preserving aspect ratio.
       - If keep_aspect_ratio is False: Resizes to exact dimensions (may distort).
    2. Else if only width is provided: Calculates height proportionally.
    3. Else if only height is provided: Calculates width proportionally.
    4. Else if max_dimension is provided: Resizes the longest side to max_dimension
       and scales the other side proportionally.
    5. If none of the above are provided, returns the original image.
    """
    original_width, original_height = image.size

    # If no arguments are provided, return the original image
    if width is None and height is None and max_dimension is None:
        return image

    new_width = original_width
    new_height = original_height

    if width is not None and height is not None:
        # No resize needed if image is already smaller than target dimensions
        if original_width <= width and original_height <= height:
            return image

        if keep_aspect_ratio:
            # Calculate the ratio to fit within the target dimensions
            ratio = min(width / original_width, height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
        else:
            # Stretch to exact dimensions
            new_width = width
            new_height = height
    elif width is not None:
        # No resize needed if width is already smaller
        if original_width <= width:
            return image
        # Calculate height proportionally
        new_width = width
        new_height = int((original_height / original_width) * new_width)
    elif height is not None:
        # No resize needed if height is already smaller
        if original_height <= height:
            return image
        # Calculate width proportionally
        new_height = height
        new_width = int((original_width / original_height) * new_height)
    elif max_dimension is not None:
        # No resize needed if both dimensions are smaller than max_dimension
        if max(original_height, original_width) <= max_dimension:
            return image

        if original_width > original_height:
            # Width is the longer side
            new_width = max_dimension
            new_height = int((original_height / original_width) * new_width)
        else:
            # Height is the longer side or sides are equal
            new_height = max_dimension
            new_width = int((original_width / original_height) * new_height)

    # Ensure dimensions are at least minimum values
    new_width = max(min_width, new_width)
    new_height = max(min_height, new_height)

    # Perform the resize operation with the calculated dimensions
    return image.resize((new_width, new_height), resample_filter)


def truncate_tokens(
    tokens: list[int],
    max_length: int,
    side: Literal["left", "middle", "right"] = "left",
) -> list[int]:
    """Truncate a token list to max_length using the given strategy (left, right, or middle)."""
    # fmt: off
    match side:
        case "left": return tokens[-max_length:]
        case "right": return tokens[:max_length]
        case "middle":
            # Truncate the middle of the sequence
            left_length = max_length // 2
            right_length = max_length - left_length
            return tokens[:left_length] + tokens[-right_length:]
        case _: raise ValueError(f"Unknown truncation {side=}. Must be one of 'left', 'middle', or 'right'.")
    # fmt: on


def maybe_truncate(
    tokens: list[int],
    max_gen_toks: int,
    max_model_len: int,
    min_gen_toks: int = 1,
    side: Literal["left", "middle", "right"] = "left",
    shrink_gen_toks=False,
    verbose=True,
) -> tuple[list[int], int]:
    """
    Truncates input tokens and/or reduces max_gen_toks to fit within max_model_len.

    Strategy:
        1. No truncation needed: If len(tokens) + max_gen_toks <= max_model_len, return as-is.
        2. If shrink_gen_toks=False: Truncate context to fit max_model_len - max_gen_toks.
        3. If shrink_gen_toks=True:
                a. First try reducing max_gen_toks (down to min_gen_toks) to fit the context.
                b. If context still doesn't fit, truncate context to reserve space for min_gen_toks.

    Args:
        tokens (list[int]): The input context tokens to potentially truncate.
        max_gen_toks (int): The maximum number of tokens to generate.
        max_model_len (int): The model's maximum context window size (prompt + generation).
        min_gen_toks (int): Lower bound for generation tokens. Defaults to 1.
        side (str): "left" | "right" | "middle". Defaults to "left".
        shrink_gen_toks (bool): Whether to adjust the generation tokens count
            to fit within the maximum length. Defaults to False.
        verbose (bool): Whether to log warnings when truncation or adjustments occur.

    Returns:
        tuple[list[int], int]: A tuple containing:
            - list[int]: The (possibly truncated) context tokens.
            - int: The adjusted maximum generation token count.

    Raises:
        ValueError: when max_model_len <= min_gen_toks.
    """
    ctx_len = len(tokens)

    # Case 1: Everything fits comfortably
    if ctx_len + max_gen_toks <= max_model_len:
        return tokens, max_gen_toks

    warning = f"Context length ({ctx_len}) + max_gen_toks ({max_gen_toks}) = {ctx_len + max_gen_toks} exceeds model's max length ({max_model_len})"

    # Case 2: Do not adjust generation tokens, just truncate prompt
    if not shrink_gen_toks:
        maybe_warn(f"{warning}. Truncating context from {side=}.", verbose)
        return truncate_tokens(
            tokens, max_model_len - max_gen_toks, side=side
        ), max_gen_toks

    # Case 3: Prompt fits, but need to reduce max_tokens
    if (new_max := max_model_len - ctx_len) >= min_gen_toks:
        maybe_warn(
            f"{warning}. Reducing {max_gen_toks=} to {new_max} to fit within model context window.",
            verbose,
        )
        return tokens, new_max

    # Case 4: Need to truncate prompt to fit min_tokens
    # Reserve space for min_tokens, use rest for prompt
    if (max_ctx_len := max_model_len - min_gen_toks) <= 0:
        raise ValueError(
            f"Model context window ({max_model_len}) is too small to fit "
            f"initial context len ({ctx_len}) + minimum generation len ({min_gen_toks})"
        )
    maybe_warn(
        f"{warning}. Truncating context from {side=} to {max_ctx_len} tokens to reserve {min_gen_toks=} for generation.",
        verbose,
    )
    return truncate_tokens(tokens, max_ctx_len, side=side), min_gen_toks


def postprocess_generated_text(
    generation: str, stop: list[str] | str | None, think_end_token: str | None
) -> str:
    """
    Post-processes the generated text by stripping stop sequences and optional thinking markers.

    Args:
        generation (str): The generated text to be processed.
        stop (list[str] | None): Stop sequence(s) to remove. Text is truncated
            at the first occurrence of any stop sequence.
        think_end_token (str | None): Token marking end of thinking section. If provided,
            returns only the text after this token (discarding thinking content).

    Returns:
        str: The processed generation - text before stop sequences and after thinking sections.
    """
    if stop:
        stop = [stop] if isinstance(stop, str) else stop
        for term in stop:
            if len(term) > 0:
                # ignore '' separator,
                # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                generation = generation.split(term)[0]
    if think_end_token:
        generation = generation.split(think_end_token)[-1].lstrip()

    return generation


def has_bos_prefix(sequence: str, bos_str: str | Iterable[str] | None = None) -> bool:
    if bos_str is None:
        return False
    elif isinstance(bos_str, str):
        return sequence.startswith(bos_str)
    else:
        return any(sequence.startswith(x) for x in bos_str)


def _add_special_kwargs(add_special_tokens: bool | None, add_bos: bool | None = None):
    if add_special_tokens is not None:
        return {"add_special_tokens": add_special_tokens}
    if add_bos is not None:
        return {"add_special_tokens": add_bos}
    return {}
