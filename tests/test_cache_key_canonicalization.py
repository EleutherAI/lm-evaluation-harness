"""Regression tests for cache-key canonicalization in `hash_args`.

`hash_args()` builds the `--use_cache` key by serializing the request arguments
with `json.dumps()`. Without `sort_keys`, the serialized bytes -- and therefore
the SHA-256 digest -- depended on the insertion order of any mapping in the
request, so two semantically identical requests built in a different order
produced two cache keys and the second one missed.

These tests pin the property that decides a key: the *content* of the request.
Mappings are order-insensitive; sequences are not, because position is part of
the content.
"""

import itertools

import pytest

from lm_eval.api.model import hash_args


def test_top_level_mapping_order_is_irrelevant():
    left = {"temperature": 0.0, "until": ["\n"]}
    right = {"until": ["\n"], "temperature": 0.0}

    assert hash_args("generate_until", ("prompt", left)) == hash_args(
        "generate_until", ("prompt", right)
    )


def test_nested_mapping_order_is_irrelevant():
    left = {"x": 0, "opts": {"alpha": 1, "beta": 2}}
    right = {"opts": {"beta": 2, "alpha": 1}, "x": 0}

    assert hash_args("generate_until", (left,)) == hash_args("generate_until", (right,))


def test_every_permutation_of_a_mapping_shares_one_key():
    fields = [("a", 1), ("b", 2), ("c", [3, 4]), ("d", {"e": 5, "f": 6})]
    keys = {hash_args("g", (dict(perm),)) for perm in itertools.permutations(fields)}
    # every key identical: order carries no information about the request
    assert len(keys) == 1


def test_sequence_order_remains_significant():
    forward = hash_args("generate_until", ({"until": ["\n", "<|endoftext|>"]},))
    reverse = hash_args("generate_until", ({"until": ["<|endoftext|>", "\n"]},))

    assert forward != reverse


def test_positional_argument_order_remains_significant():
    assert hash_args("g", ("a", "b")) != hash_args("g", ("b", "a"))


def test_attr_participates_in_the_key():
    assert hash_args("generate_until", ({"a": 1},)) != hash_args(
        "loglikelihood", ({"a": 1},)
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ({"a": 1}, {"a": 2}),
        ({"a": "1"}, {"a": 1}),
        ({"a": 1}, {"a": True}),
        ({"a": None}, {}),
        ({"a": [1, 2]}, {"a": [2, 1]}),
    ],
)
def test_content_differences_still_produce_different_keys(left, right):
    """Canonicalizing order must not collapse values that really do differ."""
    assert hash_args("g", (left,)) != hash_args("g", (right,))
