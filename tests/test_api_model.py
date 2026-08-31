import hashlib
import json
from types import SimpleNamespace

import pytest

from lm_eval import utils
from lm_eval.api.model import CachingLM, hash_args


class _FakeCache(dict):
    def commit(self):
        pass


class _FakeLM:
    def __init__(self):
        self.call_count = 0

    def generate_until(self, requests):
        self.call_count += 1
        return ["response"] * len(requests)


def _caching_lm(lm):
    caching_lm = object.__new__(CachingLM)
    caching_lm.lm = lm
    caching_lm.cache_db = "memory"
    caching_lm.dbdict = _FakeCache()
    return caching_lm


def test_hash_args_preserves_json_compatible_cache_keys():
    args = ("prompt", {"until": ["stop"], "temperature": 0})

    expected_data = json.dumps(["generate_until"] + list(args))
    expected_hash = hashlib.sha256(expected_data.encode("utf-8")).hexdigest()

    assert hash_args("generate_until", args) == expected_hash


def test_hash_args_hashes_byte_values_by_content():
    first = hash_args("generate_until", ("prompt", {}, b"image-one"))
    same = hash_args("generate_until", ("prompt", {}, b"image-one"))
    different = hash_args("generate_until", ("prompt", {}, b"image-two"))

    assert first == same
    assert first != different


def test_hash_args_distinguishes_byte_types_and_digest_strings():
    value = b"image"
    digest = hashlib.sha256(value).hexdigest()

    bytes_hash = hash_args("generate_until", (value,))
    bytearray_hash = hash_args("generate_until", (bytearray(value),))
    digest_string_hash = hash_args("generate_until", (digest,))

    assert len({bytes_hash, bytearray_hash, digest_string_hash}) == 3


def test_hash_args_hashes_pil_images_by_content():
    image_module = pytest.importorskip("PIL.Image")
    first_image = image_module.new("RGB", (2, 2), "red")
    same_image = image_module.new("RGB", (2, 2), "red")
    different_image = image_module.new("RGB", (2, 2), "blue")

    first = hash_args("generate_until", ("prompt", {}, {"visual": [first_image]}))
    same = hash_args("generate_until", ("prompt", {}, {"visual": [same_image]}))
    different = hash_args(
        "generate_until", ("prompt", {}, {"visual": [different_image]})
    )
    digest_string = hash_args(
        "generate_until",
        ("prompt", {}, {"visual": [utils.convert_pil_to_hash(first_image)]}),
    )

    assert first == same
    assert first != different
    assert first != digest_string


def test_caching_lm_reuses_multimodal_image_response():
    image_module = pytest.importorskip("PIL.Image")
    image = image_module.new("RGB", (2, 2), "red")
    request = SimpleNamespace(args=("prompt", {}, {"visual": [image]}))
    lm = _FakeLM()
    caching_lm = _caching_lm(lm)

    assert caching_lm.generate_until([request]) == ["response"]
    assert caching_lm.generate_until([request]) == ["response"]
    assert lm.call_count == 1


def test_hash_args_rejects_unsupported_objects():
    with pytest.raises(
        TypeError, match="Object of type object is not JSON serializable"
    ):
        hash_args("generate_until", (object(),))
