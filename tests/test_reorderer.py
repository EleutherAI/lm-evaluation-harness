from lm_eval.utils import Reorderer


def test_reorderer_preserves_requests_with_equal_sort_keys():
    # These likelihood requests have identical concatenated tokens but different
    # context/continuation boundaries, so they must not share a request payload.
    requests = [
        ("short", [4], [5]),
        ("two_token_continuation", [1], [2, 3]),
        ("one_token_continuation", [1, 2], [3]),
    ]

    def collate(request):
        tokens = request[1] + request[2]
        return -len(tokens), tuple(tokens)

    reorderer = Reorderer(requests, collate)
    reordered = reorderer.get_reordered()

    assert reordered == [requests[1], requests[2], requests[0]]
    assert reorderer.get_original(reordered) == requests
    assert reorderer.get_original([len(request[2]) for request in reordered]) == [
        1,
        2,
        1,
    ]
