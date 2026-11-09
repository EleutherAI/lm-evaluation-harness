import hashlib
from io import BytesIO
from unittest.mock import Mock

import pytest
from PIL import Image

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM, CachingLM, hash_args
from lm_eval.utils import convert_pil_to_hash, hash_dict_images


@pytest.fixture
def cmyk_jpeg(tmp_path):
    path = tmp_path / "image.jpg"
    Image.new("CMYK", (3, 2), (23, 87, 131, 41)).save(path, format="JPEG")
    return path


def test_cmyk_jpeg_hash_is_stable_and_preserves_input(cmyk_jpeg):
    with Image.open(cmyk_jpeg) as image, Image.open(cmyk_jpeg) as same_image:
        original_pixels = image.tobytes()
        digest = convert_pil_to_hash(image)

        assert digest == convert_pil_to_hash(same_image)
        assert digest == convert_pil_to_hash(image.copy())
        assert image.mode == "CMYK"
        assert image.tobytes() == original_pixels


def test_cmyk_hash_preserves_distinctions_lost_by_rgb_conversion():
    first = Image.new("CMYK", (3, 2), (255, 255, 255, 0))
    second = Image.new("CMYK", (3, 2), (0, 0, 0, 255))

    assert first.convert("RGB").tobytes() == second.convert("RGB").tobytes()
    assert convert_pil_to_hash(first) != convert_pil_to_hash(second)
    assert hash_args("generate_until", (first,)) != hash_args(
        "generate_until", (second,)
    )


def test_cmyk_hash_does_not_inherit_source_compression(cmyk_jpeg):
    with Image.open(cmyk_jpeg) as image:
        same_image = image.copy()
        same_image.info["compression"] = "tiff_lzw"

        assert convert_pil_to_hash(image) == convert_pil_to_hash(same_image)


def test_logged_samples_hash_cmyk_jpeg_without_replacing_original(cmyk_jpeg):
    with Image.open(cmyk_jpeg) as image:
        samples = {"task": [{"doc": {"image": image}}]}
        hashed = hash_dict_images(samples)

        assert hashed["task"][0]["doc"]["image"] == convert_pil_to_hash(image)
        assert samples["task"][0]["doc"]["image"] is image
        assert image.mode == "CMYK"


def test_response_cache_reuses_separately_decoded_cmyk_jpeg(cmyk_jpeg, tmp_path):
    lm = Mock(spec=LM)
    lm.generate_until.return_value = ["answer"]
    cache = CachingLM(lm, str(tmp_path / "responses.sqlite"))

    try:
        for _ in range(2):
            with Image.open(cmyk_jpeg) as image:
                request = Instance(
                    request_type="generate_until",
                    doc={"image": image},
                    arguments=("Question", {}, {"visual": [image]}),
                    idx=0,
                )
                assert cache.generate_until([request]) == ["answer"]
        lm.generate_until.assert_called_once()
    finally:
        cache.dbdict.close()


@pytest.mark.parametrize(
    "mode", ["1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16", "I;16B"]
)
def test_existing_png_image_hashes_remain_compatible(mode):
    image = Image.new(mode, (3, 2))
    old_encoding = BytesIO()
    image.save(old_encoding, format="PNG")
    expected = hashlib.sha256(old_encoding.getvalue()).hexdigest()

    assert convert_pil_to_hash(image) == expected
