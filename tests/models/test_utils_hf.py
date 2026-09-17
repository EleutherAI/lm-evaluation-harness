"""Tests for `lm_eval.models.utils_hf`.

These helpers are shared by the HuggingFace backends (`huggingface.py`,
`hf_vlms.py`, `hf_audiolm.py`, `mamba_lm.py`, ...) but had no direct coverage.
Everything here runs on CPU, so it executes in CI without a GPU.
"""

import pytest
import torch

from lm_eval.models.utils_hf import (
    MultiTokenEOSCriteria,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)


class StubTokenizer:
    """Minimal tokenizer stub with one token per character.

    `MultiTokenEOSCriteria` only needs `encode` and `batch_decode`. Using a
    stub keeps these tests deterministic and offline rather than downloading a
    real tokenizer in CI.
    """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inverse = {index: char for char, index in vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self.vocab[char] for char in text if char in self.vocab]

    def batch_decode(self, batch) -> list[str]:
        return [
            "".join(
                self.inverse[int(index)] for index in ids if int(index) in self.inverse
            )
            for ids in batch
        ]


@pytest.fixture()
def tokenizer() -> StubTokenizer:
    return StubTokenizer({"a": 1, "\n": 2, "b": 3})


class TestGetDtype:
    def test_converts_string_to_torch_dtype(self):
        assert get_dtype("float16") is torch.float16
        assert get_dtype("bfloat16") is torch.bfloat16
        assert get_dtype("float32") is torch.float32

    def test_auto_is_passed_through(self):
        # "auto" is resolved later against an instantiated HF AutoConfig, so it
        # must survive this helper untouched.
        assert get_dtype("auto") == "auto"

    def test_torch_dtype_is_passed_through(self):
        assert get_dtype(torch.float64) is torch.float64


class TestPadAndConcat:
    def test_right_padding(self):
        tensors = [torch.tensor([1, 2]), torch.tensor([3])]

        result = pad_and_concat(max_length=4, tensors=tensors)

        assert result.shape == (2, 4)
        assert result[0].tolist() == [1, 2, 0, 0]
        assert result[1].tolist() == [3, 0, 0, 0]

    def test_left_padding(self):
        tensors = [torch.tensor([1, 2]), torch.tensor([3])]

        result = pad_and_concat(max_length=4, tensors=tensors, padding_side="left")

        assert result.shape == (2, 4)
        assert result[0].tolist() == [0, 0, 1, 2]
        assert result[1].tolist() == [0, 0, 0, 3]

    def test_tensors_already_at_max_length_are_not_padded(self):
        tensors = [torch.tensor([1, 2, 3, 4])]

        result = pad_and_concat(max_length=4, tensors=tensors)

        assert result.shape == (1, 4)
        assert result.tolist() == [[1, 2, 3, 4]]

    def test_two_dimensional_input_is_squeezed(self):
        # Backends pass `[1, seq]` shaped tensors; the helper drops the leading
        # batch dimension so padding stays 1-D.
        tensors = [torch.tensor([[1, 2]])]

        result = pad_and_concat(max_length=3, tensors=tensors)

        assert result.shape == (1, 3)
        assert result.tolist() == [[1, 2, 0]]

    def test_unrecognized_padding_side_raises(self):
        with pytest.raises(AssertionError, match="Unrecognized padding type"):
            pad_and_concat(
                max_length=4,
                tensors=[torch.tensor([1])],
                padding_side="middle",
            )


class TestMultiTokenEOSCriteria:
    def test_returns_false_until_every_row_matches(self, tokenizer):
        criteria = MultiTokenEOSCriteria(
            sequence="\n\n",
            tokenizer=tokenizer,
            initial_decoder_input_length=0,
            batch_size=2,
        )
        # Row 0 ends in the stop sequence, row 1 does not.
        input_ids = torch.tensor([[1, 3, 2, 2], [1, 3, 1, 3]])

        assert criteria(input_ids, None) is False
        assert criteria.done_tracker == [True, False]

    def test_returns_true_once_every_row_matches(self, tokenizer):
        criteria = MultiTokenEOSCriteria(
            sequence="\n\n",
            tokenizer=tokenizer,
            initial_decoder_input_length=0,
            batch_size=2,
        )
        input_ids = torch.tensor([[1, 3, 2, 2], [1, 2, 2, 3]])

        assert criteria(input_ids, None) is True

    def test_lookback_extra_tokens_are_used(self, tokenizer):
        # The criteria looks back len(sequence) + 2 tokens so a stop sequence
        # emitted with a different tokenization is still caught.
        criteria = MultiTokenEOSCriteria(
            sequence="\n\n",
            tokenizer=tokenizer,
            initial_decoder_input_length=0,
            batch_size=1,
        )

        assert criteria.sequence_id_len == len(criteria.sequence_ids) + 2


class TestStopSequencesCriteria:
    def test_builds_one_criteria_per_sequence(self, tokenizer):
        criteria_list = stop_sequences_criteria(
            tokenizer=tokenizer,
            stop_sequences=["\n\n", "END"],
            initial_decoder_input_length=0,
            batch_size=1,
        )

        assert len(criteria_list) == 2
        assert all(
            isinstance(criteria, MultiTokenEOSCriteria) for criteria in criteria_list
        )
