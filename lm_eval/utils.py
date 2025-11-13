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
import threading
from dataclasses import asdict, is_dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple

import numpy as np
import requests
import yaml
from jinja2 import BaseLoader, Environment, StrictUndefined


SPACING = " " * 47

HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def wrap_text(string: str, width: int = 140, **kwargs) -> Optional[str]:
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


def setup_logging(verbosity=logging.INFO):
    # Configure the root logger
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.name.startswith("lm_eval."):
                record.name = record.name[len("lm_eval.") :]
            return super().format(record)

    formatter = CustomFormatter(
        "%(asctime)s %(levelname)-8s [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )

    log_level = os.environ.get("LOGLEVEL", verbosity) or verbosity

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    log_level = level_map.get(str(log_level).upper(), logging.INFO)

    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        if log_level == logging.DEBUG:
            third_party_loggers = ["urllib3", "filelock", "fsspec"]
            for logger_name in third_party_loggers:
                logging.getLogger(logger_name).setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(log_level)


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

    return re.split(r"(?<!\\)" + sep_char, text, maxsplit)


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
    if isinstance(o, np.int64) or isinstance(o, np.int32):
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


def simple_parse_args_string(args_string: Optional[str]) -> dict:
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
def pattern_match(patterns, source_list):
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


def general_detokenize(string) -> str:
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
    return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)


def sanitize_task_name(task_name: str) -> str:
    """
    Given the task name, returns a sanitized version of it.
    """
    return re.sub(r"\W", "_", task_name)


def get_latest_filename(filenames: List[str]) -> str:
    """
    Given a list of filenames, returns the filename with the latest datetime.
    """
    return max(filenames, key=lambda f: get_file_datetime(f))


def get_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to aggregated results.
    """
    return [f for f in filenames if "/results_" in f and ".json" in f]


def get_sample_results_filenames(filenames: List[str]) -> List[str]:
    """
    Extracts filenames that correspond to sample results.
    """
    return [f for f in filenames if "/samples_" in f and ".json" in f]


def get_rolling_token_windows(
    token_list: List[int], prefix_token: int, max_seq_len: int, context_len: int
) -> Generator[Tuple[List[int], List[int]], None, None]:
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
    pair: Tuple[List[int], List[int]],
) -> Tuple[List[int], List[int]]:
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
    def __init__(self, arr: List[Any], fn: Callable) -> None:
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
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = "%.4f" % v if isinstance(v, float) else v

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                se = "   N/A" if se == "N/A" else "%.4f" % se
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

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        if len(args) != 1 if inspect.ismethod(fn) else 0:
            print(
                f"WARNING: using {fn.__name__} with positional arguments is "
                "deprecated and will be disallowed in a future version of "
                "lm-evaluation-harness!"
            )
        return fn(*args, **kwargs)

    return _wrapper


def ignore_constructor(loader, node):
    return node


def import_function(loader: yaml.Loader, node, yaml_path: Path):
    function_name = loader.construct_scalar(node)

    *module_name, function_name = function_name.split(".")
    if isinstance(module_name, list):
        module_name = ".".join(module_name)
    module_path = yaml_path.parent / f"{module_name}.py"

    spec = importlib.util.spec_from_file_location(module_name, module_path.as_posix())

    if spec is None:
        raise ImportError(f"Could not import module {module_name} from {module_path}.")
    module = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Module loader is None, {module_name} from {module_path}.")
    spec.loader.exec_module(module)

    function = getattr(module, function_name)
    return function


def load_yaml_config(yaml_path=None, yaml_config=None, yaml_dir=None, mode="full"):
    if mode == "simple":
        constructor_fn = ignore_constructor
    elif mode == "full":
        if yaml_path is None:
            raise ValueError("yaml_path must be provided if mode is 'full'.")
        # Attach yaml_path to the import function so that it can be used later
        constructor_fn = functools.partial(import_function, yaml_path=Path(yaml_path))

    loader = yaml.CLoader if yaml.__with_libyaml__ else yaml.FullLoader
    # Add the import_function constructor to the YAML loader
    yaml.add_constructor("!function", constructor_fn, Loader=loader)
    if yaml_config is None:
        with open(yaml_path, "rb") as file:
            yaml_config = yaml.load(file, Loader=loader)

    if yaml_dir is None:
        yaml_dir = os.path.dirname(yaml_path)

    assert yaml_dir is not None

    if "include" in yaml_config:
        include_path = yaml_config["include"]
        del yaml_config["include"]

        if isinstance(include_path, str):
            include_path = [include_path]

        # Load from the last one first
        include_path.reverse()
        final_yaml_config = {}
        for path in include_path:
            # Assumes that path is a full path.
            # If not found, assume the included yaml
            # is in the same dir as the original yaml
            if not os.path.isfile(path):
                path = os.path.join(yaml_dir, path)

            try:
                included_yaml_config = load_yaml_config(yaml_path=path, mode=mode)
                final_yaml_config.update(included_yaml_config)
            except Exception as ex:
                # If failed to load, ignore
                raise ex

        final_yaml_config.update(yaml_config)
        return final_yaml_config
    return yaml_config


def regex_replace(string, pattern, repl, count: int = 0):
    """Implements the `re.sub` function as a custom Jinja filter."""
    return re.sub(pattern, repl, string, count=count)


env = Environment(
    loader=BaseLoader, undefined=StrictUndefined, keep_trailing_newline=True
)
env.filters["regex_replace"] = regex_replace


def apply_template(template: str, doc: dict) -> str:
    rtemplate = env.from_string(template)
    return rtemplate.render(**doc)


def create_iterator(raw_iterator, *, rank=0, world_size=1, limit=None):
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
    return hashlib.sha256(img_bytes.getvalue()).hexdigest()


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


class RemoteTokenizer:
    """
    Minimal robust tokenizer that uses vLLM server's tokenizer endpoints.
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        verify_certificate: bool = True,
        ca_cert_path: Optional[str] = None,
        auth_token: Optional[str] = None,
        max_retries: int = 3,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.RLock()
        self._tokenizer_info = None
        self._chat_template_obj = None

        # Certificate logic
        self.cert_config = (
            ca_cert_path if verify_certificate and ca_cert_path else verify_certificate
        )

        # Auth header logic
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        # Normalize base URL - remove API endpoints to get server base
        self.base_url = (
            base_url.replace("/v1/completions", "")
            .replace("/v1/chat/completions", "")
            .rstrip("/")
        )

        # Use a session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        # Validate server supports tokenizer_info endpoint
        self._validate_server()

    def _request_with_retries(self, method, url, **kwargs):
        last_exc = None
        for _ in range(self.max_retries):
            try:
                resp = self.session.request(
                    method,
                    url,
                    timeout=kwargs.pop("timeout", self.timeout),
                    verify=self.cert_config,
                    **kwargs,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_exc = e
        raise RuntimeError(
            f"RemoteTokenizer: {method} {url} failed after {self.max_retries} attempts: {last_exc}"
        )

    def _validate_server(self):
        url = f"{self.base_url}/tokenizer_info"
        resp = self._request_with_retries("GET", url)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Server does not support tokenizer_info endpoint. Status: {resp.status_code}"
            )

    @property
    def tokenizer_info(self) -> dict:
        with self._lock:
            if self._tokenizer_info is None:
                url = f"{self.base_url}/tokenizer_info"
                resp = self._request_with_retries("GET", url)
                self._tokenizer_info = resp.json()
            return self._tokenizer_info

    @property
    def eos_token(self) -> Optional[str]:
        return self.tokenizer_info.get("eos_token")

    @property
    def bos_token(self) -> Optional[str]:
        return self.tokenizer_info.get("bos_token")

    @property
    def pad_token(self) -> Optional[str]:
        return self.tokenizer_info.get("pad_token")

    @property
    def eos_token_id(self) -> Optional[int]:
        if self.eos_token is None:
            return None
        return self.encode(self.eos_token)[0]

    @property
    def bos_token_id(self) -> Optional[int]:
        if self.bos_token is None:
            return None
        return self.encode(self.bos_token)[0]

    @property
    def eot_token(self) -> Optional[int]:
        return self.eos_token_id

    def encode(self, text: str) -> List[int]:
        url = f"{self.base_url}/tokenize"
        payload = {"prompt": text, "add_special_tokens": False}
        resp = self._request_with_retries("POST", url, json=payload)
        tokens = resp.json().get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("Malformed response from /tokenize endpoint.")
        return tokens

    def decode(self, tokens: List[int]) -> str:
        url = f"{self.base_url}/detokenize"
        payload = {"tokens": tokens}
        resp = self._request_with_retries("POST", url, json=payload)
        prompt = resp.json().get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("Malformed response from /detokenize endpoint.")
        return prompt

    def batch_decode(self, tokens_list: List[List[int]]) -> List[str]:
        return [self.decode(tokens) for tokens in tokens_list]

    def apply_chat_template(
        self, chat_history: list, add_generation_prompt: bool = True, **kwargs
    ) -> str:
        with self._lock:
            if self._chat_template_obj is None:
                template_str = self.tokenizer_info.get("chat_template")
                if not template_str:
                    raise ValueError("No chat template available from server")
                self._chat_template_obj = env.from_string(template_str)
        return self._chat_template_obj.render(
            messages=chat_history, add_generation_prompt=add_generation_prompt, **kwargs
        )

    def __call__(self, text: str, add_special_tokens: bool = False, **kwargs) -> dict:
        tokens = self.encode(text)
        return {"input_ids": tokens}


def check_remote_tokenizer_support(
    base_url: str,
    timeout: int = 5,
    verify_certificate: bool = True,
    ca_cert_path: Optional[str] = None,
    auth_token: Optional[str] = None,
    max_retries: int = 3,
) -> bool:
    """
    Check if server supports remote tokenizer endpoints.
    Returns True if both /tokenizer_info and /tokenize endpoints are available and functional, False otherwise.
    """
    if not base_url:
        return False

    server_base = (
        base_url.replace("/v1/completions", "")
        .replace("/v1/chat/completions", "")
        .rstrip("/")
    )
    cert_config = (
        ca_cert_path if verify_certificate and ca_cert_path else verify_certificate
    )
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    session = requests.Session()
    session.headers.update(headers)

    def _request_with_retries(method, url, **kwargs):
        for _ in range(max_retries):
            try:
                resp = session.request(
                    method,
                    url,
                    timeout=kwargs.pop("timeout", timeout),
                    verify=cert_config,
                    **kwargs,
                )
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                pass
        return None

    # Check /tokenizer_info
    info_url = f"{server_base}/tokenizer_info"
    resp = _request_with_retries("GET", info_url)
    if not resp:
        return False
    info = resp.json()
    if not isinstance(info, dict) or "eos_token" not in info:
        return False

    # Check /tokenize
    tokenize_url = f"{server_base}/tokenize"
    test_payload = {"prompt": "test", "add_special_tokens": False}
    resp = _request_with_retries("POST", tokenize_url, json=test_payload)
    if not resp:
        return False
    tokens = resp.json().get("tokens")
    if not isinstance(tokens, list):
        return False

    return True
