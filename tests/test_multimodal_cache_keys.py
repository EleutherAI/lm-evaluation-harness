"""Regression tests for multimodal cache keys (hash_args).

`--use_cache` crashed before inference whenever a task request carried a PIL
image or byte-like payload, because `hash_args()` fed the request arguments
straight into `json.dumps()`. The fix replaces such payloads with type-tagged
content digests via a JSON `default` handler.
"""

import hashlib
import json

import pytest
from PIL import Image

from lm_eval.api.model import hash_args


def _img(color, size=(2, 2)):
    return Image.new("RGB", size, color)


def test_image_request_produces_stable_key():
    key1 = hash_args("generate_until", ("prompt", {}, {"visual": [_img((255, 0, 0))]}))
    key2 = hash_args("generate_until", ("prompt", {}, {"visual": [_img((255, 0, 0))]}))
    assert key1 == key2


def test_identical_content_same_key_across_instances():
    # separate image objects with identical content share one cache entry
    a = hash_args("generate_until", ({"visual": [_img((10, 20, 30))]},))
    b = hash_args("generate_until", ({"visual": [_img((10, 20, 30))]},))
    assert a == b


def test_different_content_different_keys():
    red = hash_args("generate_until", ({"visual": [_img((255, 0, 0))]},))
    blue = hash_args("generate_until", ({"visual": [_img((0, 0, 255))]},))
    assert red != blue


def test_bytes_payloads_hash_content():
    k = hash_args("generate_until", ({"audio": [b"payload"]},))
    assert k == hash_args("generate_until", ({"audio": [b"payload"]},))
    assert k != hash_args("generate_until", ({"audio": [b"other"]},))


def test_bytearray_matches_equivalent_bytes():
    assert hash_args("generate_until", ([b"xyz"],)) == hash_args(
        "generate_until", ([bytearray(b"xyz")],)
    )


def test_text_only_keys_remain_compatible():
    req = ("prompt", {"until": ["\n"]}, {})
    expected = hashlib.sha256(
        json.dumps(["generate_until"] + list(req)).encode("utf-8")
    ).hexdigest()
    assert hash_args("generate_until", req) == expected


def test_unsupported_types_still_fail_loudly():
    class Opaque:
        pass

    with pytest.raises(TypeError):
        hash_args("generate_until", ({"obj": Opaque()},))
