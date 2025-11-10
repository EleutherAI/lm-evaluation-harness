"""
Tests for BOS (Beginning-Of-Sequence) token handling.

Expected BOS Logic:
- add_bos_token=None: Use tokenizer's default behavior
- add_bos_token=True: Always add BOS, but never duplicate if already present
- BOS prefix detection: Check if string starts with BOS token before adding
- Loglikelihood empty context: BOS goes in context=[BOS], not continuation
- Loglikelihood with BOS in context: Don't duplicate, encode as-is

This test suite validates:
1. Defaults to None (respects tokenizer behavior)
2. No duplicate BOS (detects existing BOS prefix)
3. Chat templates work correctly (no duplication when template adds BOS)
4. Loglikelihood handles BOS correctly (reuses BOS, never duplicates)
"""

from unittest.mock import Mock

import pytest
from transformers import AutoTokenizer

from lm_eval.models.utils import _add_special_kwargs, has_bos_prefix


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def pythia_tokenizer():
    """
    Load pythia-14m tokenizer for testing.

    Properties:
    - BOS token: '<|endoftext|>' (ID: 0)
    - Does NOT add BOS by default (add_bos_token=False in tokenizer)
    - Small and fast to load
    """
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    # Set pad token to avoid padding errors in batch encoding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


@pytest.fixture(scope="module")
def gemma_tokenizer():
    """
    Load gemma-2-2b-it tokenizer for testing.

    Properties:
    - BOS token: '<bos>' (ID: 2)
    - DOES add BOS by default (add_bos_token=True in tokenizer)
    - Used to test tokenizers that add BOS by default
    """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    # Set pad token to avoid padding errors in batch encoding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# =============================================================================
# Helper Functions
# =============================================================================


def create_hf_mock(tokenizer, add_bos_token, backend="causal"):
    """Create HuggingFace model mock with tokenization methods."""
    from lm_eval.api.model import TemplateLM
    from lm_eval.models.huggingface import HFLM

    mock = Mock()
    mock.add_bos_token = add_bos_token
    mock.backend = backend
    mock.tokenizer = tokenizer
    mock.prefix_token_id = tokenizer.bos_token_id or 0
    mock.tok_encode = HFLM.tok_encode.__get__(mock, HFLM)
    mock.tok_batch_encode = HFLM.tok_batch_encode.__get__(mock, HFLM)
    mock.loglikelihood = TemplateLM.loglikelihood.__get__(mock, TemplateLM)
    mock._encode_pair = TemplateLM._encode_pair.__get__(mock, TemplateLM)
    return mock


def create_vllm_mock(tokenizer, add_bos_token):
    """Create vLLM model mock with tokenization methods."""
    from lm_eval.models.vllm_causallms import VLLM

    mock = Mock()
    mock.add_bos_token = add_bos_token
    mock.prefix_token_id = tokenizer.bos_token_id or 0
    mock.tokenizer = tokenizer
    mock.tok_encode = VLLM.tok_encode.__get__(mock, VLLM)
    return mock


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestHasBosPrefix:
    """Test BOS prefix detection utility."""

    def test_none_bos_returns_false(self):
        """When bos_token is None, should return False."""
        assert has_bos_prefix("Hello world", None) is False
        assert has_bos_prefix("<s>Hello", None) is False

    def test_detects_single_bos_string(self):
        """Should detect BOS prefix with single token string."""
        assert has_bos_prefix("<s>Hello", "<s>") is True
        assert has_bos_prefix("Hello", "<s>") is False
        assert has_bos_prefix("<s>", "<s>") is True

    def test_detects_multiple_bos_variants(self):
        """Should detect any BOS variant from a list."""
        bos_variants = ["<s>", "<bos>", "|im_start|"]
        assert has_bos_prefix("<s>Hello", bos_variants) is True
        assert has_bos_prefix("<bos>Hello", bos_variants) is True
        assert has_bos_prefix("|im_start|Hello", bos_variants) is True
        assert has_bos_prefix("Hello", bos_variants) is False


class TestAddSpecialKwargs:
    """Test add_special_tokens kwarg construction."""

    def test_explicit_add_special_tokens_takes_precedence(self):
        """Explicit add_special_tokens should override add_bos."""
        assert _add_special_kwargs(True, False) == {"add_special_tokens": True}
        assert _add_special_kwargs(False, True) == {"add_special_tokens": False}

    def test_falls_back_to_add_bos(self):
        """When add_special_tokens is None, use add_bos value."""
        assert _add_special_kwargs(None, True) == {"add_special_tokens": True}
        assert _add_special_kwargs(None, False) == {"add_special_tokens": False}

    def test_both_none_returns_empty(self):
        """When both None, return empty dict (tokenizer uses its default)."""
        assert _add_special_kwargs(None, None) == {}


# =============================================================================
# Behavior 1: Defaults to None
# =============================================================================


class TestDefaultsToNone:
    """Test that add_bos_token defaults to None, allowing tokenizer defaults."""

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_huggingface_none_uses_tokenizer_default(self, tokenizer_name, request):
        """
        HF: When add_bos_token=None, should respect tokenizer's default.

        Tests both tokenizer types:
        - Pythia: Doesn't add BOS by default
        - Gemma: DOES add BOS by default
        """
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=None)

        result = mock_hflm.tok_encode("Hello")
        expected = tokenizer.encode("Hello")
        assert result == expected

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_vllm_none_uses_tokenizer_default(self, tokenizer_name, request):
        """
        vLLM: When add_bos_token=None, should respect tokenizer's default.

        Tests both tokenizer types:
        - Pythia: Doesn't add BOS by default
        - Gemma: DOES add BOS by default
        """
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_vllm = create_vllm_mock(tokenizer, add_bos_token=None)

        result = mock_vllm.tok_encode("Hello")
        expected = tokenizer.encode("Hello")
        assert result == expected


# =============================================================================
# Behavior 2: No Duplicate BOS
# =============================================================================


class TestNoDuplicateBos:
    """Test that BOS tokens are never duplicated when already present."""

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_huggingface_detects_bos_in_single_string(self, tokenizer_name, request):
        """HF: Should detect BOS prefix and avoid duplication."""
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=True)

        bos_token = tokenizer.bos_token
        test_string = f"{bos_token}Hello"
        input_ids, _ = mock_hflm.tok_batch_encode([test_string])

        # Compare to encoding without BOS detection
        without_detection = tokenizer(
            [test_string], add_special_tokens=True, return_tensors="pt"
        )["input_ids"]

        # Check no duplicate BOS at the start
        first_token = input_ids[0][0].item()
        if first_token == tokenizer.bos_token_id:
            second_token = input_ids[0][1].item()
            assert first_token != second_token, "Should not have duplicate BOS tokens"

        # Should avoid duplication (fewer or equal tokens)
        assert input_ids.shape[1] <= without_detection.shape[1]

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_huggingface_adds_bos_when_missing(self, tokenizer_name, request):
        """HF: Should add BOS when string doesn't have it (using add_special_tokens=True)"""
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=True)

        input_ids, _ = mock_hflm.tok_batch_encode(["Hello"])
        expected = tokenizer(["Hello"], add_special_tokens=True, return_tensors="pt")[
            "input_ids"
        ]

        assert input_ids.tolist() == expected.tolist()

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_huggingface_follows_tokenizer_default(self, tokenizer_name, request):
        """
        HF: When add_bos_token is not set (None), follows tokenizer default.

        - Pythia: Doesn't add BOS by default
        - Gemma: DOES add BOS by default
        """
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=None)

        input_ids, _ = mock_hflm.tok_batch_encode(["Hello"])
        expected = tokenizer(["Hello"], return_tensors="pt")["input_ids"]

        assert input_ids.tolist() == expected.tolist()

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_vllm_handles_mixed_batch(self, tokenizer_name, add_bos_token, request):
        """
        vLLM: Should handle mixed batch (some with BOS, some without).

        Verifies correctness by comparing to expected tokenization:
        - Strings WITH BOS should not get duplicate BOS
        - Strings WITHOUT BOS behavior depends on add_bos_token setting
        """
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_vllm = create_vllm_mock(tokenizer, add_bos_token=add_bos_token)

        bos_token = tokenizer.bos_token
        batch = [f"{bos_token}Hello", "World", f"{bos_token}Foo", "Bar"]
        result = mock_vllm.tok_encode(batch)

        assert len(result) == 4

        # Verify correctness: strings WITH BOS are encoded as-is (no duplicate)
        # Strings WITHOUT BOS behavior depends on add_bos_token setting
        if add_bos_token is True:
            # Explicitly add BOS to strings without it
            world_encoding = tokenizer.encode("World", add_special_tokens=True)
            bar_encoding = tokenizer.encode("Bar", add_special_tokens=True)
        else:
            # Use tokenizer default
            world_encoding = tokenizer.encode("World")
            bar_encoding = tokenizer.encode("Bar")

        expected = [
            tokenizer.encode(f"{bos_token}Hello", add_special_tokens=False),
            world_encoding,
            tokenizer.encode(f"{bos_token}Foo", add_special_tokens=False),
            bar_encoding,
        ]

        for i, exp in enumerate(expected):
            assert result[i] == exp

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_vllm_preserves_order_in_mixed_batch(
        self, tokenizer_name, add_bos_token, request
    ):
        """vLLM: Should preserve original order after split processing."""
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_vllm = create_vllm_mock(tokenizer, add_bos_token=add_bos_token)

        bos_token = tokenizer.bos_token
        batch = [
            f"{bos_token}Apple",
            "Banana",
            "Cherry",
            f"{bos_token}Date",
            "Elderberry",
        ]
        result = mock_vllm.tok_encode(batch)

        assert len(result) == 5

        # Verify each result corresponds to the correct input
        for i, text in enumerate(batch):
            has_bos = text.startswith(bos_token)
            if has_bos:
                # Text WITH BOS: encode as-is (no duplicate)
                expected = tokenizer.encode(text, add_special_tokens=False)
            elif add_bos_token is True:
                # Text WITHOUT BOS + add_bos_token=True: explicitly add BOS
                expected = tokenizer.encode(text, add_special_tokens=True)
            else:
                # Text WITHOUT BOS + add_bos_token=None: use tokenizer default
                expected = tokenizer.encode(text)
            assert result[i] == expected


# =============================================================================
# Behavior 3: Chat Templates Work Correctly
# =============================================================================


class TestChatTemplateCompatibility:
    """Test that chat templates (which add BOS) work without duplication."""

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    def test_huggingface_chat_template_no_duplicate_bos(self, tokenizer_name, request):
        """
        HF: Chat template adds BOS, tokenizer should not add another.

        Scenario: Chat template outputs text with BOS prefix
        Expected: No duplicate BOS token in final encoding
        """
        import torch

        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=True)

        bos_token = tokenizer.bos_token
        chat_output = [f"{bos_token}User: Hello\nAssistant:"]
        input_ids, _ = mock_hflm.tok_batch_encode(chat_output)

        # Should match encoding WITHOUT add_special_tokens (no duplicate)
        expected = tokenizer(
            chat_output, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

        assert torch.equal(input_ids, expected)

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_vllm_mixed_chat_batch(self, tokenizer_name, add_bos_token, request):
        """
        vLLM: Mixed batch with chat templates should handle correctly.

        Scenario: Some messages have BOS from chat template, others don't
        Expected: Split processing, no duplicates, order preserved

        Tests both add_bos_token=None (respects tokenizer defaults) and
        add_bos_token=True (explicitly adds BOS).
        """
        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_vllm = create_vllm_mock(tokenizer, add_bos_token=add_bos_token)

        bos_token = tokenizer.bos_token
        batch = [
            f"{bos_token}System: You are helpful",
            "User: What's 2+2?",
            f"{bos_token}System: Be concise",
        ]

        result = mock_vllm.tok_encode(batch)

        assert len(result) == 3

        # Verify correctness: strings WITH BOS are encoded as-is (no duplicate)
        # Strings WITHOUT BOS behavior depends on add_bos_token setting
        if add_bos_token is True:
            # Explicitly add BOS to strings without it
            middle_encoding = tokenizer.encode(
                "User: What's 2+2?", add_special_tokens=True
            )
        else:
            # Use tokenizer default
            middle_encoding = tokenizer.encode("User: What's 2+2?")

        expected = [
            tokenizer.encode(
                f"{bos_token}System: You are helpful", add_special_tokens=False
            ),
            middle_encoding,
            tokenizer.encode(
                f"{bos_token}System: Be concise", add_special_tokens=False
            ),
        ]

        for i, exp in enumerate(expected):
            assert result[i] == exp

    def test_huggingface_seq2seq_skips_causal_bos_logic(self, pythia_tokenizer):
        """HF seq2seq: Should not apply causal-specific BOS detection."""
        mock_hflm = create_hf_mock(
            pythia_tokenizer, add_bos_token=True, backend="seq2seq"
        )

        bos_token = pythia_tokenizer.bos_token
        input_ids, _ = mock_hflm.tok_batch_encode([f"{bos_token}Hello"])

        # Should return valid tokens
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] > 0


# =============================================================================
# Behavior 4: Loglikelihood BOS Handling
# =============================================================================


class TestLoglikelihoodBosHandling:
    """Test BOS handling in loglikelihood method."""

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_empty_context_continuation_with_bos(
        self, tokenizer_name, add_bos_token, request
    ):
        """
        When context="" and continuation starts with BOS, should reuse BOS.

        Expected: (context=[BOS], continuation=[rest_of_tokens])
        Not: (context=[BOS], continuation=[BOS, rest_of_tokens])
        """
        from lm_eval.api.instance import Instance

        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=add_bos_token)
        mock_hflm.prefix_token_id = tokenizer.bos_token_id or 0
        mock_hflm._loglikelihood_tokens = lambda reqs, disable_tqdm=False: [
            (0.0, False) for _ in reqs
        ]

        bos_token = tokenizer.bos_token
        continuation = f"{bos_token}Hello"

        # Create Instance objects
        requests = [
            Instance(
                request_type="loglikelihood",
                doc={},
                arguments=("", continuation),
                idx=0,
            )
        ]

        # Call loglikelihood and capture what gets passed to _loglikelihood_tokens
        captured_reqs = []

        def capture_and_return(reqs, disable_tqdm=False):
            captured_reqs.extend(reqs)
            return [(0.0, False) for _ in reqs]

        mock_hflm._loglikelihood_tokens = capture_and_return
        mock_hflm.loglikelihood(requests)

        # Verify tokenization
        _, context_enc, continuation_enc = captured_reqs[0]

        # Context should be just BOS
        assert context_enc == [tokenizer.bos_token_id]

        # Continuation should NOT include BOS (it was moved to context)
        continuation_without_bos = tokenizer.encode(
            continuation, add_special_tokens=False
        )
        assert continuation_enc == continuation_without_bos[1:]  # Skip the BOS token

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_empty_context_continuation_without_bos(
        self, tokenizer_name, add_bos_token, request
    ):
        """
        When context="" and continuation doesn't start with BOS, should add BOS as context.

        Expected: (context=[BOS], continuation=[full_continuation_tokens])
        """
        from lm_eval.api.instance import Instance

        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=add_bos_token)
        mock_hflm.prefix_token_id = tokenizer.bos_token_id or 0

        continuation = "Hello"
        requests = [
            Instance(
                request_type="loglikelihood",
                doc={},
                arguments=("", continuation),
                idx=0,
            )
        ]

        captured_reqs = []

        def capture_and_return(reqs, disable_tqdm=False):
            captured_reqs.extend(reqs)
            return [(0.0, False) for _ in reqs]

        mock_hflm._loglikelihood_tokens = capture_and_return
        mock_hflm.loglikelihood(requests)

        _, context_enc, continuation_enc = captured_reqs[0]

        # Context should be BOS
        assert context_enc == [tokenizer.bos_token_id]

        # Continuation should be full continuation (no BOS in original)
        expected_continuation = tokenizer.encode(continuation, add_special_tokens=False)
        assert continuation_enc == expected_continuation

    @pytest.mark.parametrize("tokenizer_name", ["pythia_tokenizer", "gemma_tokenizer"])
    @pytest.mark.parametrize("add_bos_token", [None, True])
    def test_context_with_bos_prefix(self, tokenizer_name, add_bos_token, request):
        """When context starts with BOS (e.g., from chat template), should not duplicate BOS."""
        from lm_eval.api.instance import Instance

        tokenizer = request.getfixturevalue(tokenizer_name)
        mock_hflm = create_hf_mock(tokenizer, add_bos_token=add_bos_token)
        mock_hflm.prefix_token_id = tokenizer.bos_token_id or 0

        bos_token = tokenizer.bos_token
        requests = [
            Instance(
                request_type="loglikelihood",
                doc={},
                arguments=(f"{bos_token}System:", "Hello"),
                idx=0,
            )
        ]

        captured_reqs = []
        mock_hflm._loglikelihood_tokens = lambda reqs, disable_tqdm=False: (
            captured_reqs.extend(reqs) or [(0.0, False) for _ in reqs]
        )
        mock_hflm.loglikelihood(requests)

        _, context_enc, continuation_enc = captured_reqs[0]

        # Verify no duplicate BOS and correct structure
        assert context_enc[0] == tokenizer.bos_token_id, "Context should start with BOS"
        assert len(context_enc) > 1, "Context should have content beyond BOS"
        assert context_enc[1] != tokenizer.bos_token_id, "Should not have duplicate BOS"
        assert continuation_enc[0] != tokenizer.bos_token_id, (
            "Continuation should not start with BOS"
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_explicit_override_takes_precedence(self, pythia_tokenizer):
        """Explicit add_special_tokens should override add_bos_token."""
        mock_hflm = create_hf_mock(pythia_tokenizer, add_bos_token=True)

        # Explicitly override to False
        result = mock_hflm.tok_encode("Hello", add_special_tokens=False)

        # Should match tokenizer with add_special_tokens=False
        expected = pythia_tokenizer.encode("Hello", add_special_tokens=False)
        assert result == expected

    def test_vllm_empty_input(self):
        """vLLM should handle empty input gracefully."""
        from lm_eval.models.vllm_causallms import VLLM

        mock_vllm = Mock()
        mock_vllm.tok_encode = VLLM.tok_encode.__get__(mock_vllm, VLLM)

        assert mock_vllm.tok_encode("") == []
        assert mock_vllm.tok_encode([]) == []
