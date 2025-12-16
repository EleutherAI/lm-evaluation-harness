import itertools

import numpy as np
import pytest
import torch

from lm_eval.api.metrics import (
    aggregate_subtask_metrics,
    mean,
    pooled_sample_stderr,
    stderr_for_metric,
)
from lm_eval.models.utils import Collator
from lm_eval.utils import (
    RemoteTokenizer,
    check_remote_tokenizer_support,
    get_rolling_token_windows,
    make_disjoint_window,
)


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v1():
    gold = [
        ([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ),
        (
            [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        ),
        ([23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [30, 31, 32, 33]),
    ]
    x = list(range(34))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=10,
        context_len=1,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v2():
    gold = [
        ([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [10, 11, 12]),
        ([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [13, 14, 15]),
        ([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [16, 17, 18]),
        ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [19, 20, 21]),
        ([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [22, 23, 24]),
        ([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [25, 26, 27]),
        ([20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [28, 29, 30]),
        ([23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [31, 32, 33]),
    ]
    x = list(range(34))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=10,
        context_len=8,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v3():
    gold = [
        ([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]),
        ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12]),
        ([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13]),
        ([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [14]),
        ([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15]),
        ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16]),
        ([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [17]),
        ([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [18]),
        ([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [19]),
        ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20]),
        ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21]),
        ([12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22]),
        ([13, 14, 15, 16, 17, 18, 19, 20, 21, 22], [23]),
        ([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24]),
        ([15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [25]),
        ([16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [26]),
        ([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [27]),
        ([18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [28]),
        ([19, 20, 21, 22, 23, 24, 25, 26, 27, 28], [29]),
        ([20, 21, 22, 23, 24, 25, 26, 27, 28, 29], [30]),
        ([21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [31]),
        ([22, 23, 24, 25, 26, 27, 28, 29, 30, 31], [32]),
        ([23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [33]),
    ]
    x = list(range(34))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=10,
        context_len=10,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v4():
    gold = [
        ([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11]),
        ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [12]),
        ([3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13]),
        ([4, 5, 6, 7, 8, 9, 10, 11, 12, 13], [14]),
        ([5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [15]),
        ([6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16]),
        ([7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [17]),
        ([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [18]),
        ([9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [19]),
        ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [20]),
        ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [21]),
        ([12, 13, 14, 15, 16, 17, 18, 19, 20, 21], [22]),
        ([13, 14, 15, 16, 17, 18, 19, 20, 21, 22], [23]),
        ([14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24]),
        ([15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [25]),
        ([16, 17, 18, 19, 20, 21, 22, 23, 24, 25], [26]),
        ([17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [27]),
        ([18, 19, 20, 21, 22, 23, 24, 25, 26, 27], [28]),
        ([19, 20, 21, 22, 23, 24, 25, 26, 27, 28], [29]),
    ]
    x = list(range(30))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=10,
        context_len=10,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v5():
    gold = [
        ([-100, 0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (
            [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ),
        (
            [19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        ),
    ]
    x = list(range(30))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=10,
        context_len=1,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


# noinspection DuplicatedCode
def test_get_rolling_token_windows_v6():
    gold = [
        ([-100, 0], [0, 1]),
        ([1, 2], [2, 3]),
        ([3, 4], [4, 5]),
        ([5, 6], [6, 7]),
        ([6, 7], [8]),
    ]
    x = list(range(9))
    generator = get_rolling_token_windows(
        token_list=x,
        prefix_token=-100,
        max_seq_len=2,
        context_len=1,
    )
    pred_length = 0
    output = []
    for input_tokens, pred_tokens in generator:
        output.extend([(input_tokens, pred_tokens)])
        pred_length += len(pred_tokens)
    assert pred_length == len(x)
    assert gold == output


def test_get_rolling_token_windows_empty():
    generator = get_rolling_token_windows(
        token_list=[],
        prefix_token=-100,
        max_seq_len=2,
        context_len=1,
    )
    n = 0
    for _ in generator:
        n += 1
    assert n == 0


def test_make_disjoint_window():
    assert make_disjoint_window(([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])) == (
        [1],
        [2, 3, 4, 5, 6],
    )
    assert make_disjoint_window(([1, 2, 3, 4, 5], [4, 5, 6])) == ([1, 2, 3], [4, 5, 6])
    assert make_disjoint_window(([1, 2, 3, 4, 5], [6])) == ([1, 2, 3, 4, 5], [6])


class TestCollator:
    def make_generate_sample(self, end=10):
        strings = ["x" * i for i in range(1, end + 1)]
        gen_kwargs1, gen_kwargs2 = (
            {"temperature": 0},
            {"temperature": 0, "until": ["nn", "\n\n"]},
        )
        args = [
            (string, gen_kwargs1 if i < len(strings) // 2 else gen_kwargs2)
            for i, string in enumerate(strings)
        ]

        return args

    def make_loglikelihood_sample(self, end=11):
        samples = [
            (("x", "x"), list(range(1, total_length + 1)))
            for total_length in range(1, end + 1)
        ]
        return samples

    def make_loglikelihood_sample_group(self, end=11):
        a = [(("x", "x"), [1, 2, 3, 4, 5, 6, 7, 8], [x]) for x in range(9)]
        b = [
            (("x", "x"), [1, 2, 3, 4, 5, 6, 7, 8], [x, y, z])
            for x, y, z in zip(range(9), range(9, 18), range(18, 27))
        ]
        return a + b

    @pytest.mark.parametrize("batch_size, end", [(17, 30), (8, 61), (12, 48), (0, 9)])
    def test_generations(self, batch_size, end):
        _collate_gen = lambda x: (-len(x[0]), x[0])  # noqa: E731

        generation_samples = self.make_generate_sample(int(end))
        gens = Collator(generation_samples, _collate_gen, group_by="gen_kwargs")
        chunks_gen = gens.get_batched(n=int(batch_size), batch_fn=None)
        output = []
        group_one = end // 2
        group_two = end - end // 2
        is_batch = batch_size != 0
        for chunks in chunks_gen:
            # check batching
            assert (
                len(chunks) <= batch_size
                if is_batch
                else len(chunks) in [group_one, group_two]
            )
            # check if reorder-er is working correctly
            chunk_lengths = [len(chunk[0]) for chunk in chunks]
            assert chunk_lengths == sorted(chunk_lengths, reverse=True)
            # check if grouping correctly
            chunk_to_compare = chunks[0][1]
            assert all(x[1] == chunk_to_compare for x in chunks)
            for x in chunks:
                output.extend([x])
        reordered_output = gens.get_original(output)
        # check get original
        assert reordered_output == generation_samples

    @pytest.mark.parametrize("batch_size, end", [(17, 30), (8, 61), (12, 48), (0, 3)])
    def test_loglikelihood(self, batch_size, end):
        _collate_log = lambda x: (-len(x[1]), tuple(x[1]))  # noqa: E731
        loglikelihood_samples = self.make_loglikelihood_sample(int(end))
        loglikelihoods = Collator(
            loglikelihood_samples,
            _collate_log,
        )
        chunks_gen = loglikelihoods.get_batched(n=int(batch_size), batch_fn=None)
        output = []
        is_batch = batch_size != 0
        for chunks in chunks_gen:
            # check batching
            assert len(chunks) <= batch_size if is_batch else len(chunks) == end
            # check reorder
            chunk_lengths = [len(chunk[1]) for chunk in chunks]
            assert chunk_lengths == sorted(chunk_lengths, reverse=True)
            for x in chunks:
                output.extend([x[1]])
        # check indices
        reordered_output = loglikelihoods.get_original(output)
        assert reordered_output == [x[1] for x in loglikelihood_samples]

    @pytest.mark.parametrize("batch_size", [17, 8, 12, 0])
    def test_context_grouping(self, batch_size):
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        _collate_log = _collate  # noqa: E731
        loglikelihood_samples = self.make_loglikelihood_sample_group()
        loglikelihoods = Collator(
            loglikelihood_samples,
            _collate_log,
            group_fn=lambda a: a[-2] + a[-1][:-1],
            group_by="contexts",
        )
        chunks_gen = loglikelihoods.get_batched(n=int(batch_size), batch_fn=None)
        output = []
        outputs_ = []
        is_batch = batch_size != 0
        for chunks in chunks_gen:
            # check batching
            if is_batch:
                assert len(chunks) <= batch_size
            # check reorder
            chunk_lengths = [len(chunk[1]) for chunk in chunks]
            assert chunk_lengths == sorted(chunk_lengths, reverse=True)
            for x in chunks:
                for request_str, cont_toks, logits in loglikelihoods.get_cache(
                    req_str="".join(x[0]),
                    cxt_toks=x[1],
                    cont_toks=x[2],
                    logits=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
                    .unsqueeze(0)
                    .unsqueeze(0),
                ):
                    output.extend([x[1]])
                    outputs_.extend([cont_toks])
        assert len(output) == len(outputs_)
        # check indices
        reordered_output = loglikelihoods.get_original(output)
        assert reordered_output == [x[1] for x in loglikelihood_samples]


def test_aggregate_mean():
    # test weight_by_size is respected
    assert (
        aggregate_subtask_metrics([0.3, 0.2, 0.4], [20, 40, 100], weight_by_size=False)
        == 0.3
    )
    assert (
        aggregate_subtask_metrics([0.3, 0.2, 0.4], [20, 40, 100], weight_by_size=True)
        == 0.3375
    )


@pytest.mark.parametrize(
    "samples",
    [
        [40 * [1.0] + 60 * [0.0], 30 * [1.0] + 30 * [0.0], 20 * [1.0] + 60 * [0.0]],
        [35 * [1.0] + 65 * [0.0], 20 * [1.0] + 20 * [0.0]],
    ],
)
def test_aggregate_stderrs(samples):
    # check that aggregating subtasks' bootstrap stderrs with our formula
    # (using weight_by_size) is ~equiv.
    # to just getting bootstrap stderr of the whole set of samples
    mean_stderr = stderr_for_metric(metric=mean, bootstrap_iters=100000)

    stderrs = [mean_stderr(subtask) for subtask in samples]

    sizes = [len(subtask) for subtask in samples]

    assert np.allclose(
        pooled_sample_stderr(stderrs, sizes),
        mean_stderr(list(itertools.chain.from_iterable(samples))),
        atol=1.0e-3,
    )


def test_remote_tokenizer_custom_cert_and_token(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {
                "name_or_path": "mock",
                "chat_template": "{{ messages[0].content }}",
            }

        def raise_for_status(self):
            pass

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "requests.Session.request", lambda self, method, url, **kwargs: DummyResponse()
    )
    tokenizer = RemoteTokenizer(
        base_url="https://mock-server",
        verify_certificate=True,
        ca_cert_path="dummy.crt",
        auth_token="dummy-token",
    )
    assert tokenizer.cert_config == "dummy.crt"
    assert tokenizer.headers["Authorization"] == "Bearer dummy-token"
    assert tokenizer.tokenizer_info["name_or_path"] == "mock"


def test_remote_tokenizer_no_cert(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {"name_or_path": "mock"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "requests.Session.request", lambda self, method, url, **kwargs: DummyResponse()
    )
    tokenizer = RemoteTokenizer(
        base_url="https://mock-server",
        verify_certificate=True,
        ca_cert_path=None,
        auth_token="dummy-token",
    )
    assert tokenizer.cert_config is True
    assert tokenizer.headers["Authorization"] == "Bearer dummy-token"
    assert tokenizer.tokenizer_info["name_or_path"] == "mock"


def test_remote_tokenizer_http_url(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {"name_or_path": "mock"}

        def raise_for_status(self):
            pass

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "requests.Session.request", lambda self, method, url, **kwargs: DummyResponse()
    )
    tokenizer = RemoteTokenizer(
        base_url="http://mock-server",
        verify_certificate=True,
        ca_cert_path="dummy.crt",
        auth_token="dummy-token",
    )
    assert tokenizer.base_url.startswith("http://")
    assert tokenizer.tokenizer_info["name_or_path"] == "mock"


def test_check_remote_tokenizer_support(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return self._json

        def raise_for_status(self):
            pass

        def __init__(self, url, json=None):
            if "tokenizer_info" in url:
                self._json = {
                    "name_or_path": "mock",
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                    "pad_token": "<pad>",
                    "chat_template": "{{ messages[0].content }}",
                }
            elif "tokenize" in url:
                self._json = {"tokens": [1, 2, 3]}
            else:
                self._json = {}

    monkeypatch.setattr("os.path.exists", lambda path: True)

    def dummy_request(self, method, url, **kwargs):
        return DummyResponse(url, json=kwargs.get("json"))

    monkeypatch.setattr("requests.Session.request", dummy_request)
    assert check_remote_tokenizer_support(
        base_url="https://mock-server",
        verify_certificate=True,
        ca_cert_path="dummy.crt",
        auth_token="dummy-token",
    )


def test_apply_chat_template(monkeypatch):
    class DummyResponse:
        status_code = 200

        def json(self):
            return {
                "name_or_path": "mock",
                "chat_template": "{{ messages[0].content }}",
            }

        def raise_for_status(self):
            pass

    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        "requests.Session.request", lambda self, method, url, **kwargs: DummyResponse()
    )
    tokenizer = RemoteTokenizer(
        base_url="https://mock-server",
        verify_certificate=True,
        ca_cert_path="dummy.crt",
        auth_token="dummy-token",
    )
    chat_history = [{"role": "user", "content": "Hello"}]
    rendered = tokenizer.apply_chat_template(chat_history)
    assert rendered == "Hello"


# Tests for lm_eval.api.utils
from lm_eval.api.utils import maybe_delimit, requires_delimiter


class TestRequiresDelimiter:
    """Tests for requires_delimiter function."""

    def test_no_whitespace_requires_delimiter(self):
        """Neither string has whitespace at boundary - delimiter required."""
        assert requires_delimiter("hello", "world") is True

    def test_prefix_ends_with_space(self):
        """Prefix ends with space - no delimiter needed."""
        assert requires_delimiter("hello ", "world") is False

    def test_suffix_starts_with_space(self):
        """Suffix starts with space - no delimiter needed."""
        assert requires_delimiter("hello", " world") is False

    def test_both_have_whitespace(self):
        """Both have whitespace at boundary - no delimiter needed."""
        assert requires_delimiter("hello ", " world") is False

    def test_prefix_ends_with_newline(self):
        """Prefix ends with newline - no delimiter needed."""
        assert requires_delimiter("hello\n", "world") is False

    def test_suffix_starts_with_tab(self):
        """Suffix starts with tab - no delimiter needed."""
        assert requires_delimiter("hello", "\tworld") is False


class TestMaybeDelimit:
    """Tests for maybe_delimit function."""

    def test_both_present_no_whitespace(self):
        """Both strings present, neither has whitespace - adds delimiter."""
        assert maybe_delimit("hello", "world") == "hello world"

    def test_both_present_prefix_has_space(self):
        """Prefix ends with space - no delimiter added."""
        assert maybe_delimit("hello ", "world") == "hello world"

    def test_both_present_suffix_has_space(self):
        """Suffix starts with space - no delimiter added."""
        assert maybe_delimit("hello", " world") == "hello world"

    def test_custom_delimiter(self):
        """Custom delimiter is used when needed."""
        assert maybe_delimit("hello", "world", delimiter="-") == "hello-world"

    def test_prefix_is_none(self):
        """Prefix is None - returns suffix."""
        assert maybe_delimit(None, "world") == "world"

    def test_prefix_is_empty(self):
        """Prefix is empty string - returns suffix."""
        assert maybe_delimit("", "world") == "world"

    def test_suffix_is_none(self):
        """Suffix is None - returns prefix."""
        assert maybe_delimit("hello", None) == "hello"

    def test_suffix_is_empty(self):
        """Suffix is empty string - returns prefix."""
        assert maybe_delimit("hello", "") == "hello"

    def test_both_none(self):
        """Both are None - returns empty string."""
        assert maybe_delimit(None, None) == ""

    def test_both_empty(self):
        """Both are empty strings - returns empty string."""
        assert maybe_delimit("", "") == ""

    def test_newline_delimiter(self):
        """Newline delimiter is used correctly."""
        assert maybe_delimit("line1", "line2", delimiter="\n") == "line1\nline2"

    def test_prefix_ends_with_newline_no_extra_delimiter(self):
        """Prefix ends with newline - no extra delimiter added."""
        assert maybe_delimit("line1\n", "line2", delimiter=" ") == "line1\nline2"
