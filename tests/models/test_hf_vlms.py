from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from PIL import Image

from lm_eval.api.registry import get_model
from lm_eval.models.hf_vlms import DEFAULT_IMAGE_PLACEHOLDER, HFMultimodalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import flatten_image_list, replace_placeholders, resize_image


def test_hf_multimodal_registered_and_subclass() -> None:
    """Verify registry lookup, base subclass, and class attributes."""
    assert get_model("hf-multimodal") is HFMultimodalLM
    assert issubclass(HFMultimodalLM, HFLM)
    assert HFMultimodalLM.MULTIMODAL is True
    assert HFMultimodalLM.AUTO_MODEL_CLASS is transformers.AutoModelForImageTextToText


@pytest.mark.parametrize(
    ("kwargs", "expected_err"),
    [
        (
            {"image_max_side": 512, "image_width": 256},
            "Ambiguous config for image resize",
        ),
        (
            {"image_max_side": 512, "image_height": 256},
            "Ambiguous config for image resize",
        ),
    ],
)
def test_hf_multimodal_rejects_ambiguous_resize_args(
    kwargs: dict[str, int], expected_err: str
) -> None:
    """Verify ValueError when both image_max_side and explicit width/height are set."""
    with pytest.raises(ValueError, match=expected_err):
        HFMultimodalLM.__new__(HFMultimodalLM).__init__(
            pretrained="mock-model",
            **kwargs,
        )


def test_hf_multimodal_rejects_auto_batch_size() -> None:
    """Verify assertion error when batch_size='auto' is requested."""
    with (
        patch.object(HFLM, "__init__", return_value=None),
        pytest.raises(AssertionError, match="Batch size 'auto' is not yet supported"),
    ):
        lm = HFMultimodalLM.__new__(HFMultimodalLM)
        lm.batch_size_per_gpu = "auto"
        lm._config = SimpleNamespace()
        lm.__init__(pretrained="mock-model")


def test_hf_multimodal_image_token_resolution() -> None:
    """Test resolution of image_token from string, token_id, and config attributes."""
    # 1. Missing all identifiers raises AssertionError
    with (
        patch.object(HFLM, "__init__", return_value=None),
        pytest.raises(AssertionError, match="Must have a non-None image_token_id"),
    ):
        lm = HFMultimodalLM.__new__(HFMultimodalLM)
        lm.batch_size_per_gpu = 1
        lm._config = SimpleNamespace()
        lm.__init__(pretrained="mock-model")

    # 2. Explicit image_string takes precedence and sets image_token directly
    with patch.object(HFLM, "__init__", return_value=None):
        lm = HFMultimodalLM.__new__(HFMultimodalLM)
        lm.batch_size_per_gpu = 1
        lm._config = SimpleNamespace()
        lm.__init__(pretrained="mock-model", image_string="<custom_image_tag>")
        assert lm.image_token == "<custom_image_tag>"  # noqa: S105

    # 3. Explicit image_token_id decodes token via tok_decode
    with patch.object(HFLM, "__init__", return_value=None):
        lm = HFMultimodalLM.__new__(HFMultimodalLM)
        lm.batch_size_per_gpu = 1
        lm._config = SimpleNamespace()
        lm.tok_decode = MagicMock(return_value="<decoded_image>")
        lm.__init__(pretrained="mock-model", image_token_id=32000)
        assert lm.image_token_id == 32000
        assert lm.image_token == "<decoded_image>"  # noqa: S105
        lm.tok_decode.assert_called_once_with([32000], skip_special_tokens=False)

    # 4. Fallback to config.image_token_index
    with patch.object(HFLM, "__init__", return_value=None):
        lm = HFMultimodalLM.__new__(HFMultimodalLM)
        lm.batch_size_per_gpu = 1
        lm._config = SimpleNamespace(image_token_index=32001)
        lm.tok_decode = MagicMock(return_value="<fallback_token>")
        lm.__init__(pretrained="mock-model")
        assert lm.image_token_id == 32001
        assert lm.image_token == "<fallback_token>"  # noqa: S105


def test_hf_multimodal_apply_chat_template_interleaved() -> None:
    """Verify chat template transformation with interleaved images and max_images cap."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    lm.interleave = True
    lm.max_images = 2
    lm.processor = MagicMock()
    lm.processor.apply_chat_template.side_effect = lambda msgs, **kwargs: (
        "rendered_chat"
    )

    chat_history = [
        {
            "role": "user",
            "content": f"First {DEFAULT_IMAGE_PLACEHOLDER} then {DEFAULT_IMAGE_PLACEHOLDER} end.",
        }
    ]

    result = lm.apply_chat_template(chat_history, add_generation_prompt=True)
    assert result == "rendered_chat"
    assert lm.chat_applied is True

    # Check structural conversion
    converted_content = chat_history[0]["content"]
    assert converted_content == [
        {"type": "text", "text": "First "},
        {"type": "image"},
        {"type": "text", "text": " then "},
        {"type": "image"},
        {"type": "text", "text": " end."},
    ]


def test_hf_multimodal_apply_chat_template_non_interleaved() -> None:
    """Verify chat template transformation when interleave=False prepends image blocks."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    lm.interleave = False
    lm.max_images = 1
    lm.processor = MagicMock()
    lm.processor.apply_chat_template.return_value = "rendered_non_interleaved"

    # Input contains 2 placeholders, but max_images=1
    chat_history = [
        {
            "role": "user",
            "content": f"Compare {DEFAULT_IMAGE_PLACEHOLDER} and {DEFAULT_IMAGE_PLACEHOLDER}.",
        }
    ]

    result = lm.apply_chat_template(chat_history, add_generation_prompt=False)
    assert result == "rendered_non_interleaved"
    assert lm.chat_applied is True

    converted_content = chat_history[0]["content"]
    assert converted_content == [
        {"type": "image", "image": None},
        {"type": "text", "text": "Compare  and ."},
    ]


def test_hf_multimodal_encode_pair_whitespace_handling() -> None:
    """Verify trailing context whitespace is shifted to continuation prior to tokenization."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    dummy_img = Image.new("RGB", (10, 10))

    # Mock tok_multimodal_encode to record inputs and return predictable token IDs
    def mock_encode(
        text: str, images: list[Image.Image]
    ) -> tuple[list[list[int]], dict]:
        tokens = [ord(c) for c in text]
        return [tokens], {"pixel_values": "mock_pixels"}

    lm.tok_multimodal_encode = MagicMock(side_effect=mock_encode)

    context = "Prompt with spaces   "
    continuation = "Answer"
    ctx_enc, cont_enc, img_enc = lm._encode_multimodal_pair(
        context, continuation, [dummy_img]
    )

    # First call encodes context + continuation with spaces shifted
    # "Prompt with spaces" + "   Answer"
    assert lm.tok_multimodal_encode.call_args_list[0][0][0] == (
        "Prompt with spaces   Answer"
    )
    # Second call encodes trimmed context
    assert lm.tok_multimodal_encode.call_args_list[1][0][0] == ("Prompt with spaces")

    expected_ctx = [ord(c) for c in "Prompt with spaces"]
    expected_cont = [ord(c) for c in "   Answer"]
    assert ctx_enc == expected_ctx
    assert cont_enc == expected_cont
    assert img_enc == {"pixel_values": "mock_pixels"}


def test_hf_multimodal_tok_multimodal_encode_left_truncate() -> None:
    """Verify left_truncate_len correctly slices output tokens."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    processor = MagicMock()
    processor.return_value = {
        "input_ids": [10, 20, 30, 40, 50],
        "attention_mask": [1, 1, 1, 1, 1],
        "pixel_values": torch.zeros((1, 3, 224, 224)),
    }
    lm.processor = processor

    text_enc, extra_enc = lm.tok_multimodal_encode(
        string="sample query", images=[], left_truncate_len=3
    )

    assert text_enc == [30, 40, 50]
    assert "input_ids" not in extra_enc
    assert "attention_mask" not in extra_enc
    assert "pixel_values" in extra_enc


def test_hf_multimodal_tok_batch_encode_llava_and_rgb() -> None:
    """Verify batch tokenization flattens images for llava and converts RGB mode."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    lm.chat_applied = False
    lm.image_token = "<custom_img>"  # noqa: S105
    lm.max_images = 2
    lm.rgb = True
    lm._config = SimpleNamespace(model_type="llava")
    lm._device = torch.device("cpu")
    lm._model = SimpleNamespace(dtype=torch.float32)

    tokenizer = SimpleNamespace(padding_side="right")
    lm.tokenizer = tokenizer

    processor = MagicMock()
    mock_batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
    }
    mock_batch_encoding = MagicMock()
    mock_batch_encoding.__getitem__.side_effect = mock_batch.__getitem__
    mock_batch_encoding.__setitem__.side_effect = mock_batch.__setitem__
    processor.return_value = mock_batch_encoding
    lm.processor = processor

    gray_img1 = Image.new("L", (10, 10))
    gray_img2 = Image.new("L", (10, 10))

    _ = lm.tok_batch_multimodal_encode(
        strings=["Query <image>"],
        images=[[gray_img1, gray_img2]],
        padding_side="left",
    )

    assert tokenizer.padding_side == "right"  # Restores padding side
    processor.assert_called_once()
    called_kwargs = processor.call_args.kwargs
    # For llava, images are flattened to a 1D list of PIL Images
    assert isinstance(called_kwargs["images"], list)
    assert all(isinstance(img, Image.Image) for img in called_kwargs["images"])
    # All images were converted to RGB
    assert all(img.mode == "RGB" for img in called_kwargs["images"])
    # Placeholder replaced
    assert called_kwargs["text"] == ["Query <custom_img>"]


def test_hf_multimodal_model_call_no_grad() -> None:
    """Verify _model_multimodal_call evaluates under torch.no_grad and passes image kwargs."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    mock_model = MagicMock()
    mock_model.return_value = SimpleNamespace(logits=torch.tensor([[[0.1, 0.9]]]))
    lm._model = mock_model

    inps = torch.tensor([[1, 2]])
    imgs = {"pixel_values": torch.zeros((1, 3, 224, 224))}

    logits = lm._model_multimodal_call(inps, imgs)
    assert torch.is_tensor(logits)
    assert logits.shape == (1, 1, 2)
    mock_model.assert_called_once_with(inps, **imgs)


def test_image_utility_functions() -> None:
    """Verify auxiliary image manipulation functions: resize, flatten, replace."""
    # 1. flatten_image_list
    img1, img2, img3 = (
        Image.new("RGB", (5, 5)),
        Image.new("RGB", (5, 5)),
        Image.new("RGB", (5, 5)),
    )
    flattened = flatten_image_list([[img1], [img2, img3]])
    assert flattened == [img1, img2, img3]

    # 2. replace_placeholders
    raw_text = "Img 1: <image>, Img 2: <image>, Img 3: <image>"
    replaced = replace_placeholders(raw_text, "<image>", "<tok>", max_images=2)
    assert replaced == "Img 1: <tok>, Img 2: <tok>, Img 3: <image>"

    # 3. resize_image explicit dimensions (keep_aspect_ratio=False for exact size)
    large_img = Image.new("RGB", (100, 100))
    resized_exact = resize_image(
        large_img, width=20, height=30, keep_aspect_ratio=False
    )
    assert resized_exact.size == (20, 30)

    # 4. resize_image max_dimension preserving aspect ratio
    wide_img = Image.new("RGB", (100, 50))
    resized_side = resize_image(wide_img, max_dimension=50)
    assert resized_side.size == (50, 25)
