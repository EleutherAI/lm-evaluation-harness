from lm_eval.utils import get_rolling_token_windows, make_disjoint_window


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
