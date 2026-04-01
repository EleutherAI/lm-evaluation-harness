"""Tests for correct context length handling in vLLM.

- _loglikelihood_tokens: max_cxt_len = max_length - 1 (vLLM requires at least 1 generation token)
- loglikelihood_rolling: max_seq_len = max_length - 2 (1 for context, 1 for generation)
"""

from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.skip(reason="requires vLLM, not available in CI")

vllm = pytest.importorskip("vllm")

from lm_eval.models.vllm_causallms import VLLM


class TestVLLMContextLength:
    """Tests for vLLM context length truncation in _loglikelihood_tokens."""

    def test_loglikelihood_tokens_truncates_to_max_length_minus_one(self) -> None:
        """Test that _loglikelihood_tokens truncates inputs to max_length - 1."""
        # Create a mock VLLM instance with required attributes
        mock_lm = MagicMock(spec=VLLM)
        mock_lm.max_length = 10
        mock_lm.batch_size = 1
        mock_lm.rank = 0
        mock_lm.cache_hook = MagicMock()

        # Capture what gets passed to _model_generate
        captured_inputs = []

        def capture_model_generate(requests, generate):
            captured_inputs.extend(requests)
            # Return mock outputs with prompt_logprobs
            mock_outputs = []
            for req in requests:
                mock_output = MagicMock()
                # prompt_logprobs: first is None, rest are dicts mapping token -> logprob
                mock_output.prompt_logprobs = [None] + [
                    {token: MagicMock(logprob=-0.5)} for token in req[1:]
                ]
                mock_outputs.append(mock_output)
            return mock_outputs

        mock_lm._model_generate = capture_model_generate

        # Context: 6 tokens, continuation: 6 tokens = 12 total (exceeds max_length of 10)
        context_enc = [1, 2, 3, 4, 5, 6]
        continuation_enc = [7, 8, 9, 10, 11, 12]
        requests = [(("ctx", "cont"), context_enc, continuation_enc)]

        # Call the actual method on the mock instance
        VLLM._loglikelihood_tokens(mock_lm, requests, disable_tqdm=True)

        # Should truncate to max_length - 1 = 9 tokens
        assert len(captured_inputs) == 1
        assert len(captured_inputs[0]) == 9, (
            f"Expected 9 tokens (max_length - 1), got {len(captured_inputs[0])}"
        )
        # Should keep the last 9 tokens
        assert captured_inputs[0] == [4, 5, 6, 7, 8, 9, 10, 11, 12]

    def test_loglikelihood_tokens_no_truncation_when_within_limit(self) -> None:
        """Test that _loglikelihood_tokens doesn't truncate when input fits."""
        mock_lm = MagicMock(spec=VLLM)
        mock_lm.max_length = 10
        mock_lm.batch_size = 1
        mock_lm.rank = 0
        mock_lm.cache_hook = MagicMock()

        captured_inputs = []

        def capture_model_generate(requests, generate):
            captured_inputs.extend(requests)
            mock_outputs = []
            for req in requests:
                mock_output = MagicMock()
                mock_output.prompt_logprobs = [None] + [
                    {token: MagicMock(logprob=-0.5)} for token in req[1:]
                ]
                mock_outputs.append(mock_output)
            return mock_outputs

        mock_lm._model_generate = capture_model_generate

        # Context: 4 tokens, continuation: 4 tokens = 8 total (within max_length - 1 = 9)
        context_enc = [1, 2, 3, 4]
        continuation_enc = [5, 6, 7, 8]
        requests = [(("ctx", "cont"), context_enc, continuation_enc)]

        VLLM._loglikelihood_tokens(mock_lm, requests, disable_tqdm=True)

        # Should not truncate - 8 tokens < 9 (max_length - 1)
        assert len(captured_inputs) == 1
        assert len(captured_inputs[0]) == 8, (
            f"Expected 8 tokens (no truncation), got {len(captured_inputs[0])}"
        )
        assert captured_inputs[0] == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_loglikelihood_tokens_truncates_at_exactly_max_length(self) -> None:
        """Test that input of exactly max_length gets truncated to max_length - 1."""
        mock_lm = MagicMock(spec=VLLM)
        mock_lm.max_length = 10
        mock_lm.batch_size = 1
        mock_lm.rank = 0
        mock_lm.cache_hook = MagicMock()

        captured_inputs = []

        def capture_model_generate(requests, generate):
            captured_inputs.extend(requests)
            mock_outputs = []
            for req in requests:
                mock_output = MagicMock()
                mock_output.prompt_logprobs = [None] + [
                    {token: MagicMock(logprob=-0.5)} for token in req[1:]
                ]
                mock_outputs.append(mock_output)
            return mock_outputs

        mock_lm._model_generate = capture_model_generate

        # Context: 5 tokens, continuation: 5 tokens = 10 total (exactly max_length)
        context_enc = [1, 2, 3, 4, 5]
        continuation_enc = [6, 7, 8, 9, 10]
        requests = [(("ctx", "cont"), context_enc, continuation_enc)]

        VLLM._loglikelihood_tokens(mock_lm, requests, disable_tqdm=True)

        # Should truncate to 9 tokens (max_length - 1) because vLLM needs 1 generation token
        assert len(captured_inputs) == 1
        assert len(captured_inputs[0]) == 9, (
            f"Expected 9 tokens (max_length - 1), got {len(captured_inputs[0])}"
        )
        assert captured_inputs[0] == [2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_loglikelihood_tokens_boundary_at_max_length_minus_one(self) -> None:
        """Test boundary case where input is exactly max_length - 1 (no truncation needed)."""
        mock_lm = MagicMock(spec=VLLM)
        mock_lm.max_length = 10
        mock_lm.batch_size = 1
        mock_lm.rank = 0
        mock_lm.cache_hook = MagicMock()

        captured_inputs = []

        def capture_model_generate(requests, generate):
            captured_inputs.extend(requests)
            mock_outputs = []
            for req in requests:
                mock_output = MagicMock()
                mock_output.prompt_logprobs = [None] + [
                    {token: MagicMock(logprob=-0.5)} for token in req[1:]
                ]
                mock_outputs.append(mock_output)
            return mock_outputs

        mock_lm._model_generate = capture_model_generate

        # Context: 5 tokens, continuation: 4 tokens = 9 total (exactly max_length - 1)
        context_enc = [1, 2, 3, 4, 5]
        continuation_enc = [6, 7, 8, 9]
        requests = [(("ctx", "cont"), context_enc, continuation_enc)]

        VLLM._loglikelihood_tokens(mock_lm, requests, disable_tqdm=True)

        # Should NOT truncate - exactly at the limit
        assert len(captured_inputs) == 1
        assert len(captured_inputs[0]) == 9, (
            f"Expected 9 tokens (at boundary, no truncation), got {len(captured_inputs[0])}"
        )
        assert captured_inputs[0] == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_loglikelihood_rolling_uses_max_length_minus_two(self) -> None:
        """Test that loglikelihood_rolling creates windows with max_seq_len = max_length - 2.

        Rolling needs room for:
        - 1 token for context (the prefix token)
        - 1 token for generation (required by vLLM)
        So max_seq_len = max_length - 2
        """
        mock_lm = MagicMock(spec=VLLM)
        mock_lm.max_length = 10
        mock_lm.batch_size = 1
        mock_lm.rank = 0
        mock_lm.prefix_token_id = 0
        mock_lm.cache_hook = MagicMock()

        # Mock tok_encode to return a long sequence that will be split into windows
        # With max_length=10 and max_seq_len=8, windows should be at most 8 tokens
        mock_lm.tok_encode = MagicMock(return_value=list(range(1, 21)))  # 20 tokens

        # Capture the windows passed to _loglikelihood_tokens
        captured_windows = []

        def capture_loglikelihood_tokens(requests, disable_tqdm):
            captured_windows.extend(requests)
            # Return mock results (logprob, is_greedy) for each request
            return [(-1.0, True) for _ in requests]

        mock_lm._loglikelihood_tokens = capture_loglikelihood_tokens

        # Create a mock request
        mock_request = MagicMock()
        mock_request.args = ("test string",)

        VLLM.loglikelihood_rolling(mock_lm, [mock_request], disable_tqdm=True)

        # Verify windows were created
        assert len(captured_windows) > 0, "Expected windows to be created"

        # Each window is (cache_key, context_enc, continuation_enc)
        # With max_seq_len = max_length - 2 = 8, after make_disjoint_window,
        # total = max_seq_len + 1 = 9, which fits vLLM's limit of max_length - 1 = 9
        for window in captured_windows:
            cache_key, context_enc, continuation_enc = window
            total_len = len(context_enc) + len(continuation_enc)
            # Windows from rolling should be at most max_length - 1 = 9
            # This is the vLLM limit (needs 1 token for generation)
            assert total_len <= 9, (
                f"Window too large: {total_len} tokens, expected <= 9 (max_length - 1)"
            )
            # Also verify it's NOT larger than max_length - 1 (which would fail in _loglikelihood_tokens)
