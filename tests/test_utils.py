import pytest

from lm_eval.utils import (
    Collator,
    calculate_combined_variance,
    get_rolling_token_windows,
    make_disjoint_window,
    weighted_mean,
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
        output.append((input_tokens, pred_tokens))
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
        output.append((input_tokens, pred_tokens))
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
        output.append((input_tokens, pred_tokens))
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
        output.append((input_tokens, pred_tokens))
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
        output.append((input_tokens, pred_tokens))
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
        output.append((input_tokens, pred_tokens))
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

    @pytest.mark.parametrize("batch_size, end", [(17, 30), (8, 61), (12, 48), (0, 9)])
    def test_generations(self, batch_size, end):
        _collate_gen = lambda x: (-len(x[0]), x[0])  # noqa: E731

        generation_samples = self.make_generate_sample(int(end))
        gens = Collator(generation_samples, _collate_gen, grouping=True)
        chunks = gens.get_batched(n=int(batch_size), batch_fn=None)
        output = []
        for chunks in chunks:
            # check batching
            group_one = end // 2
            group_two = end - end // 2
            assert (
                len(chunks) <= batch_size
                if batch_size != 0
                else len(chunks) in [group_one, group_two]
            )
            # check if reorder-er is working correctly
            assert all(
                len(chunks[i][0]) <= len(chunks[i - 1][0])
                for i in range(1, len(chunks))
            )
            # check if grouping correctly
            assert all(x[1] == chunks[0][1] for x in chunks)
            for x in chunks:
                output.append(x)
        reordered_output = gens.get_original(output)
        # check get original
        assert reordered_output == generation_samples

    @pytest.mark.parametrize("batch_size, end", [(17, 30), (8, 61), (12, 48), (0, 3)])
    def test_loglikelihood(self, batch_size, end):
        _collate_log = lambda x: (-len(x[1]), tuple(x[1]))  # noqa: E731
        loglikelihood_samples = self.make_loglikelihood_sample(int(end))
        loglikelihoods = Collator(loglikelihood_samples, _collate_log, grouping=False)
        chunks = loglikelihoods.get_batched(n=int(batch_size), batch_fn=None)
        output = []
        for chunks in chunks:
            # check batching
            assert len(chunks) <= batch_size if batch_size != 0 else len(chunks) == end
            # check reorder
            assert all(
                len(chunks[i][1]) <= len(chunks[i - 1][1])
                for i in range(1, len(chunks))
            )
            for x in chunks:
                output.append(x[1])
        # check indices
        reordered_output = loglikelihoods.get_original(output)
        assert reordered_output == [x[1] for x in loglikelihood_samples]

    def test_calculate_combined_variance(self):
        datasets = [
            [6, 30.66667, 8.555555],
            [7, 31.14286, 13.26531],
            [3, 40.33333, 1.555555],
        ]
        current_n = 0
        current_mean = 0
        current_variance = 0
        for n, mean, variance in datasets:
            if current_n == 0:
                current_n, current_mean, current_variance = n, mean, variance
            else:
                new_n = current_n + n
                new_mean = weighted_mean(current_n, n, current_mean, mean)
                current_variance = calculate_combined_variance(
                    current_n, n, current_variance, variance, current_mean, mean
                )
                current_n, current_mean = new_n, new_mean

        assert current_variance == pytest.approx(22.804053)
