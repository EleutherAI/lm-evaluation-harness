from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from PIL import Image
from transformers import BatchFeature

from lm_eval.api.instance import Instance
from lm_eval.api.model import CachingLM, hash_args
from lm_eval.api.registry import get_model
from lm_eval.models.hf_vlms import DEFAULT_IMAGE_PLACEHOLDER, HFMultimodalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import flatten_image_list, replace_placeholders, resize_image


if TYPE_CHECKING:
    from pathlib import Path


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


class _FakeCharTokenizer:
    """Minimal deterministic tokenizer for offline HFMultimodalLM end-to-end tests."""

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    IMAGE_TOKEN: str = "<|img|>"  # noqa: S105
    IMAGE_TOKEN_ID: int = 3
    vocab_size: int = 512

    def __init__(self) -> None:
        self.padding_side: str = "right"
        self.chat_template: str | dict[str, str] | None = "<tok_chat>"
        self.default_chat_template: str | None = None

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs: object,
    ) -> list[int]:
        """Encode text deterministically, preserving special image and EOS markers."""
        tokens: list[int] = [self.bos_token_id] if add_special_tokens else []
        idx = 0
        while idx < len(text):
            if text.startswith(self.IMAGE_TOKEN, idx):
                tokens.append(self.IMAGE_TOKEN_ID)
                idx += len(self.IMAGE_TOKEN)
            elif text.startswith("</s>", idx):
                tokens.append(self.eos_token_id)
                idx += 4
            else:
                tokens.append(ord(text[idx]) + 10)
                idx += 1
        return tokens

    def decode(
        self,
        token_ids: int | list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            ids = [int(x) for x in token_ids.tolist()]
        elif isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = [int(x) for x in token_ids]

        special_map = {
            self.pad_token_id: "<pad>",
            self.bos_token_id: "<s>",
            self.eos_token_id: "</s>",
            self.IMAGE_TOKEN_ID: self.IMAGE_TOKEN,
        }
        chars: list[str] = []
        for tid in ids:
            if tid in special_map:
                if not skip_special_tokens or tid == self.IMAGE_TOKEN_ID:
                    chars.append(special_map[tid])
            elif tid >= 10:
                chars.append(chr(tid - 10))
        return "".join(chars)

    def batch_decode(self, sequences: list[list[int]] | torch.Tensor) -> list[str]:
        """Decode a batch of token sequences while retaining special tokens."""
        rows = sequences.tolist() if isinstance(sequences, torch.Tensor) else sequences
        return [self.decode(row, skip_special_tokens=False) for row in rows]


class _FakeVLMProcessor:
    """Minimal deterministic processor returning real BatchFeature tensors."""

    def __init__(self) -> None:
        self.tokenizer = _FakeCharTokenizer()
        self.calls: list[dict[str, object]] = []
        self.chat_template: str | dict[str, str] | None = {
            "default": "<proc_default>",
            "rag": "<proc_rag>",
        }
        self.default_chat_template: str | None = None

    def apply_chat_template(
        self, messages: list[dict[str, object]], **kwargs: object
    ) -> str:
        """Render chat messages into a synthetic prompt string."""
        return "rendered"

    def __call__(
        self,
        images: object = None,
        text: str | list[str] = "",
        truncation: bool = False,
        padding: str | bool = False,
        return_tensors: str | None = None,
    ) -> BatchFeature | dict[str, object]:
        """Record call metadata and encode text alongside synthetic pixel tensors."""
        self.calls.append(
            {
                "images": images,
                "text": text,
                "truncation": truncation,
                "padding": padding,
                "return_tensors": return_tensors,
                "padding_side": self.tokenizer.padding_side,
            }
        )
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [self.tokenizer.encode(item) for item in texts]
        if return_tensors != "pt":
            return {
                "input_ids": encoded,
                "attention_mask": [[1] * len(seq) for seq in encoded],
                "pixel_values": torch.ones((len(texts), 3, 4, 4), dtype=torch.float32),
            }

        max_len = max((len(seq) for seq in encoded), default=0)
        padded_ids: list[list[int]] = []
        padded_mask: list[list[int]] = []
        left_pad = self.tokenizer.padding_side == "left"
        for seq in encoded:
            pad_len = max_len - len(seq)
            if left_pad:
                padded_ids.append([self.tokenizer.pad_token_id] * pad_len + seq)
                padded_mask.append([0] * pad_len + [1] * len(seq))
            else:
                padded_ids.append(seq + [self.tokenizer.pad_token_id] * pad_len)
                padded_mask.append([1] * len(seq) + [0] * pad_len)

        return BatchFeature(
            {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
                "pixel_values": torch.ones((len(texts), 3, 4, 4), dtype=torch.float32),
            }
        )


class _FakeVLMModel:
    """Minimal pre-initialized VLM stub compatible with HFLM and HFMultimodalLM."""

    def __init__(
        self,
        tokenizer: _FakeCharTokenizer,
        reply_by_prompt_substr: dict[str, str] | None = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.device = torch.device("cpu")
        self.dtype = dtype
        self.name_or_path = "offline/fake-vlm"
        self.config = SimpleNamespace(
            model_type="qwen2_vl",
            image_token_id=tokenizer.IMAGE_TOKEN_ID,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        self._tokenizer = tokenizer
        self.reply_by_prompt_substr = (
            dict(reply_by_prompt_substr) if reply_by_prompt_substr is not None else {}
        )
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Append deterministic reply tokens matched by prompt substring."""
        self.generate_calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        assert isinstance(input_ids, torch.Tensor)
        encoded_replies: list[list[int]] = []
        for row in input_ids:
            prompt_text = self._tokenizer.decode(row, skip_special_tokens=True)
            reply = ""
            for substr, candidate in self.reply_by_prompt_substr.items():
                if substr in prompt_text:
                    reply = candidate
                    break
            rep_ids = self._tokenizer.encode(reply)
            if self._tokenizer.eos_token_id in rep_ids:
                eos_idx = rep_ids.index(self._tokenizer.eos_token_id)
                rep_ids = rep_ids[: eos_idx + 1]
            encoded_replies.append(rep_ids)

        max_reply = max((len(rep) for rep in encoded_replies), default=0)
        padded_replies = [
            rep + [self._tokenizer.pad_token_id] * (max_reply - len(rep))
            for rep in encoded_replies
        ]
        suffix = torch.tensor(
            padded_replies, dtype=input_ids.dtype, device=input_ids.device
        )
        return torch.cat([input_ids, suffix], dim=1)


def _build_offline_vlm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reply_by_prompt_substr: dict[str, str] | None = None,
    batch_size: int = 2,
    max_length: int = 64,
    image_max_side: int | None = None,
    convert_img_format: bool = False,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[HFMultimodalLM, _FakeVLMProcessor, _FakeVLMModel]:
    """Construct HFMultimodalLM via real __init__ using offline doubles."""
    processor = _FakeVLMProcessor()
    model = _FakeVLMModel(
        processor.tokenizer, reply_by_prompt_substr=reply_by_prompt_substr
    )
    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: processor,
    )
    lm = HFMultimodalLM(
        pretrained=model,
        backend="causal",
        batch_size=batch_size,
        max_length=max_length,
        image_max_side=image_max_side,
        convert_img_format=convert_img_format,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return lm, processor, model


def test_hf_multimodal_full_init_and_processor_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify full HFMultimodalLM.__init__ resolves processor, pixels, and image token."""
    processor = _FakeVLMProcessor()
    model = _FakeVLMModel(processor.tokenizer)
    captured: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_from_pretrained(*args: object, **kwargs: object) -> _FakeVLMProcessor:
        captured.append((args, kwargs))
        return processor

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", fake_from_pretrained
    )

    lm = HFMultimodalLM(
        pretrained=model,
        backend="causal",
        batch_size=2,
        max_length=64,
        min_pixels=128,
        max_pixels=1024,
        convert_img_format=True,
    )

    assert captured == [
        (
            ("offline/fake-vlm",),
            {
                "revision": "main",
                "trust_remote_code": False,
                "min_pixels": 128,
                "max_pixels": 1024,
            },
        )
    ]
    assert lm.processor is processor
    assert lm.tokenizer is processor.tokenizer
    assert lm.image_token_id == processor.tokenizer.IMAGE_TOKEN_ID
    assert lm.image_token == processor.tokenizer.IMAGE_TOKEN
    assert lm.rgb is True
    assert lm.batch_size == 2
    assert lm.max_length == 64


def test_hf_multimodal_generate_until_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify generate_until end-to-end batching, image resize, stop trimming, and cache."""
    lm, processor, model = _build_offline_vlm(
        monkeypatch,
        reply_by_prompt_substr={
            "Short": "Alpha\nStop:ignored",
            "Much longer": "Beta</s>extra",
        },
        batch_size=2,
        max_length=64,
        image_max_side=40,
        convert_img_format=True,
    )

    shared_gen_kwargs: dict[str, object] = {"until": ["\nStop:"], "max_gen_toks": 12}
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(
                f"Short {DEFAULT_IMAGE_PLACEHOLDER}",
                shared_gen_kwargs,
                {"visual": [Image.new("L", (100, 50))]},
            ),
            idx=0,
        ),
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(
                f"Much longer prompt with {DEFAULT_IMAGE_PLACEHOLDER} context",
                shared_gen_kwargs,
                {"visual": [Image.new("L", (80, 40))]},
            ),
            idx=1,
        ),
    ]

    caching_lm = CachingLM(lm, str(tmp_path / "cache.db"))
    with contextlib.closing(caching_lm.dbdict):
        outputs = lm.generate_until(requests, disable_tqdm=True)

        assert outputs == ["Alpha", "Beta"]
        assert shared_gen_kwargs == {"until": ["\nStop:"], "max_gen_toks": 12}

        assert len(processor.calls) == 1
        proc_call = processor.calls[0]
        assert isinstance(proc_call["text"], list)
        # Collator sorts longest context first before restoring original request order
        assert proc_call["text"] == [
            f"Much longer prompt with {processor.tokenizer.IMAGE_TOKEN} context",
            f"Short {processor.tokenizer.IMAGE_TOKEN}",
        ]
        sent_images = proc_call["images"]
        assert isinstance(sent_images, list)
        assert len(sent_images) == 2
        for img_group in sent_images:
            assert isinstance(img_group, list)
            assert len(img_group) == 1
            assert img_group[0].mode == "RGB"
            assert img_group[0].size == (40, 20)

        assert len(model.generate_calls) == 1
        gen_call = model.generate_calls[0]
        assert isinstance(gen_call["pixel_values"], torch.Tensor)
        assert gen_call["pixel_values"].dtype == torch.float16
        assert isinstance(gen_call["input_ids"], torch.Tensor)
        assert gen_call["input_ids"].dtype == torch.long

        assert (
            caching_lm.dbdict[
                hash_args(
                    "generate_until",
                    (f"Short {DEFAULT_IMAGE_PLACEHOLDER}", shared_gen_kwargs),
                )
            ]
            == "Alpha"
        )
        assert (
            caching_lm.dbdict[
                hash_args(
                    "generate_until",
                    (
                        f"Much longer prompt with {DEFAULT_IMAGE_PLACEHOLDER} context",
                        shared_gen_kwargs,
                    ),
                )
            ]
            == "Beta"
        )


def test_hf_multimodal_generate_until_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify generate_until handles empty batches and rejects non-dict gen_kwargs."""
    lm, processor, model = _build_offline_vlm(monkeypatch, batch_size=1, max_length=64)

    assert lm.generate_until([], disable_tqdm=True) == []
    assert processor.calls == []
    assert model.generate_calls == []

    bad_req = Instance(
        request_type="generate_until",
        doc={},
        arguments=(
            f"Ask {DEFAULT_IMAGE_PLACEHOLDER}",
            "invalid_kwargs",
            {"visual": [Image.new("RGB", (8, 8))]},
        ),
        idx=0,
    )
    with pytest.raises(ValueError, match="Expected `kwargs` to be of type `dict`"):
        lm.generate_until([bad_req], disable_tqdm=True)


def test_hf_multimodal_model_generate_kwargs_and_stop_criteria(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify greedy/sampling temperature normalization and MultiTokenEOSCriteria wiring."""
    lm, processor, model = _build_offline_vlm(monkeypatch)
    prompt_ids = torch.tensor([processor.tokenizer.encode("Hi ")], dtype=torch.long)
    inputs = {"input_ids": prompt_ids}

    lm._model_multimodal_generate(inputs, max_length=32, stop=["END"], temperature=0.0)
    greedy_call = model.generate_calls[0]
    assert "temperature" not in greedy_call
    assert greedy_call["do_sample"] is False
    assert greedy_call["use_cache"] is True
    assert greedy_call["pad_token_id"] == processor.tokenizer.pad_token_id

    stopping_criteria = greedy_call["stopping_criteria"]
    assert isinstance(stopping_criteria, transformers.StoppingCriteriaList)
    seq_no_stop = torch.tensor([processor.tokenizer.encode("Hi there")])
    seq_with_stop = torch.tensor([processor.tokenizer.encode("Hi doneEND")])
    assert not bool(stopping_criteria(seq_no_stop, None))
    assert bool(stopping_criteria(seq_with_stop, None))

    lm._model_multimodal_generate(
        inputs, max_length=32, stop=["END"], temperature=0.7, do_sample=True
    )
    sample_call = model.generate_calls[1]
    assert sample_call["temperature"] == 0.7
    assert sample_call["do_sample"] is True


def test_hf_multimodal_chat_template_selection() -> None:
    """Verify chat_template delegates to processor when supported and restores tokenizer."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    tokenizer = _FakeCharTokenizer()
    processor = _FakeVLMProcessor()
    lm.tokenizer = tokenizer
    lm.processor = processor

    assert lm.chat_template("rag") == "<proc_rag>"
    assert lm.tokenizer is tokenizer
    assert lm.chat_template(True) == "<proc_default>"
    assert lm.tokenizer is tokenizer

    # Processor without apply_chat_template falls back to tokenizer.chat_template
    lm.processor = SimpleNamespace()
    assert lm.chat_template(True) == "<tok_chat>"
    assert lm.tokenizer is tokenizer


def test_hf_multimodal_tok_batch_encode_chat_applied_and_truncate() -> None:
    """Verify chat_applied skips tag replacement, keeps nested images, and truncates."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    processor = _FakeVLMProcessor()
    lm.processor = processor
    lm.tokenizer = processor.tokenizer
    lm.chat_applied = True
    lm.image_token = processor.tokenizer.IMAGE_TOKEN
    lm.max_images = 1
    lm.rgb = False
    lm._config = SimpleNamespace(model_type="qwen2_vl")
    lm._device = torch.device("cpu")
    lm._model = SimpleNamespace(dtype=torch.float16)

    img1 = Image.new("RGB", (8, 8))
    img2 = Image.new("RGB", (8, 8))
    encoding = lm.tok_batch_multimodal_encode(
        strings=[f"abcd {DEFAULT_IMAGE_PLACEHOLDER}"],
        images=[[img1, img2]],
        padding_side="left",
        left_truncate_len=2,
    )

    assert len(processor.calls) == 1
    call = processor.calls[0]
    assert call["text"] == [f"abcd {DEFAULT_IMAGE_PLACEHOLDER}"]
    assert call["images"] == [[img1]]
    assert encoding["input_ids"].shape == (1, 2)
    assert encoding["attention_mask"].shape == (1, 2)
    assert encoding["pixel_values"].dtype == torch.float16
    assert processor.tokenizer.padding_side == "right"


def test_hf_multimodal_batch_images_and_request_routing() -> None:
    """Verify _batch_images concatenation and multimodal vs text-only routing."""
    lm = HFMultimodalLM.__new__(HFMultimodalLM)
    lm._device = torch.device("cpu")
    lm._model = SimpleNamespace(dtype=torch.float16)

    batched = lm._batch_images(
        [
            {"pixel_values": [[[1.0, 2.0]]], "image_grid_thw": [[1, 2, 2]]},
            {"pixel_values": [[[3.0, 4.0]]], "image_grid_thw": [[1, 4, 4]]},
        ]
    )
    assert batched["pixel_values"].shape == (2, 1, 2)
    assert batched["pixel_values"].dtype == torch.float16
    assert torch.equal(
        batched["pixel_values"],
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float16),
    )
    assert batched["image_grid_thw"].shape == (2, 3)

    img_aux = {"visual": [Image.new("RGB", (4, 4))]}
    mm_ll_req = Instance(
        request_type="loglikelihood",
        doc={},
        arguments=(f"ctx {DEFAULT_IMAGE_PLACEHOLDER}", " cont", img_aux),
        idx=0,
    )
    with pytest.raises(NotImplementedError, match="not yet tested"):
        lm.loglikelihood([mm_ll_req])

    mm_roll_req = Instance(
        request_type="loglikelihood_rolling",
        doc={},
        arguments=(f"ctx {DEFAULT_IMAGE_PLACEHOLDER}", {"until": []}, img_aux),
        idx=0,
    )
    with pytest.raises(
        NotImplementedError, match="does not support loglikelihood_rolling"
    ):
        lm.loglikelihood_rolling([mm_roll_req])

    text_ll_req = Instance(
        request_type="loglikelihood", doc={}, arguments=("ctx", " cont"), idx=0
    )
    text_roll_req = Instance(
        request_type="loglikelihood_rolling", doc={}, arguments=("text only",), idx=0
    )
    text_gen_req = Instance(
        request_type="generate_until",
        doc={},
        arguments=("prompt", {"until": []}),
        idx=0,
    )

    with patch.object(HFLM, "loglikelihood", return_value=[(-0.5, True)]) as ll_mock:
        assert lm.loglikelihood([text_ll_req], disable_tqdm=True) == [(-0.5, True)]
        ll_mock.assert_called_once_with(requests=[text_ll_req], disable_tqdm=True)

    with patch.object(HFLM, "loglikelihood_rolling", return_value=[-1.25]) as roll_mock:
        assert lm.loglikelihood_rolling([text_roll_req]) == [-1.25]
        roll_mock.assert_called_once_with(requests=[text_roll_req])

    with patch.object(HFLM, "generate_until", return_value=["ok"]) as gen_mock:
        assert lm.generate_until([text_gen_req], disable_tqdm=True) == ["ok"]
        gen_mock.assert_called_once_with(requests=[text_gen_req], disable_tqdm=True)
