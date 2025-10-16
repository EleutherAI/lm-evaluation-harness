"""
Test suite for TemplateLM class from lm_eval.api.model

This file provides boilerplate mocking and test fixtures for testing
the TemplateLM abstract base class methods.
"""

from __future__ import annotations

import os
import random
from typing import Optional
from unittest.mock import Mock, patch

import pytest

from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Mock TemplateLM Implementation
# ============================================================================


class MockTemplateLM(TemplateLM):
    """
    Concrete implementation of TemplateLM for testing purposes.
    Override abstract methods with mock implementations.
    """

    def __init__(
        self,
        tokenizer=None,
        eot_token_id: int = 0,
        prefix_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self._eot_token_id = eot_token_id
        self._prefix_token_id = prefix_token_id
        self.AUTO_MODEL_CLASS = None  # Set to specific class in tests if needed

    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id

    @property
    def prefix_token_id(self) -> int:
        if self._prefix_token_id is not None:
            return self._prefix_token_id
        return self.eot_token_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        """Mock tokenization - override in tests as needed"""
        # Use tokenizer if available, otherwise fall back to character codes
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            result = self.tokenizer.encode(string, **kwargs)
            # Handle both list returns and Mock returns
            return result if isinstance(result, list) else list(result)
        # Fallback: return list of character codes
        return [ord(c) for c in string]

    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        """Mock implementation - override in tests"""
        return requests

    def loglikelihood_rolling(
        self, requests, disable_tqdm: bool = False
    ) -> list[float]:
        """Mock implementation - override in tests"""
        return [-1.0 for _ in requests]

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        """Mock implementation - override in tests"""
        return ["mock_output" for _ in requests]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_tokenizer():
    """Provides a mock tokenizer object with common attributes"""
    tokenizer = Mock()
    tokenizer.chat_template = None
    tokenizer.default_chat_template = "default template"
    # Mock encode to return token IDs (using character codes as simple token IDs)
    tokenizer.encode = Mock(
        side_effect=lambda x, **kwargs: [0] + [ord(c) for c in x]
        if kwargs.get("add_special_tokens")
        else [ord(c) for c in x]
    )
    # Mock decode to convert token IDs back to string
    tokenizer.decode = Mock(side_effect=lambda x, **kwargs: "".join(chr(c) for c in x))
    return tokenizer


@pytest.fixture
def mock_lm():
    """Provides a basic MockTemplateLM instance"""
    return MockTemplateLM(eot_token_id=0)


@pytest.fixture
def mock_lm_bos():
    """Provides a MockTemplateLM instance with a BOS token"""
    return MockTemplateLM(eot_token_id=0, prefix_token_id=1)


@pytest.fixture
def mock_lm_with_tokenizer(mock_tokenizer):
    """Provides a MockTemplateLM instance with a tokenizer"""
    return MockTemplateLM(tokenizer=mock_tokenizer, eot_token_id=0)


@pytest.fixture
def sample_instances():
    """Provides sample Instance objects for testing"""
    return [
        Instance(
            request_type="loglikelihood",
            doc={"context": "test", "continuation": "output"},
            arguments=("context1", "continuation1"),
            idx=0,
        ),
        Instance(
            request_type="loglikelihood",
            doc={"context": "test2", "continuation": "output2"},
            arguments=("context2", "continuation2"),
            idx=1,
        ),
    ]


@pytest.fixture
def sample_empty_context_instance():
    """Provides an Instance with empty context for testing edge cases"""
    return Instance(
        request_type="loglikelihood",
        doc={"context": "", "continuation": "output"},
        arguments=("", "continuation"),
        idx=0,
    )


# ============================================================================
# Test Class
# ============================================================================


class TestTemplateLM:
    """Test suite for TemplateLM methods"""

    # ------------------------------------------------------------------------
    # Property Tests
    # ------------------------------------------------------------------------

    def test_eot_token_id(self, mock_lm):
        """Test eot_token_id property"""
        assert mock_lm.eot_token_id == 0

    def test_prefix_token_id_default(self, mock_lm):
        """Test that prefix_token_id defaults to eot_token_id"""
        assert mock_lm.prefix_token_id == mock_lm.eot_token_id

    def test_prefix_token_id_custom(self, mock_lm_bos):
        """Test custom prefix_token_id"""
        assert mock_lm_bos.prefix_token_id == 1

    # ------------------------------------------------------------------------
    # tok_encode Tests
    # ------------------------------------------------------------------------

    def test_tok_encode_empty_string(self, mock_lm, mock_tokenizer):
        """Test tok_encode with empty string"""
        mock_lm.tokenizer = mock_tokenizer
        with pytest.raises(AssertionError):
            mock_lm._encode_pair("", "hello")

    # ------------------------------------------------------------------------
    # _encode_pair Tests
    # ------------------------------------------------------------------------

    def test_encode_pair(self, mock_lm, sample_instances, mock_tokenizer):
        """Test tok_encode with a simple string"""
        mock_lm.tokenizer = mock_tokenizer
        for instance in sample_instances:
            context, cont = instance.args
            context_enc, cont_enc = mock_lm._encode_pair(context, cont)
            assert context == mock_lm.tokenizer.decode(context_enc)
            assert cont == mock_lm.tokenizer.decode(cont_enc)

    def test_encode_pair_context_trailing_spaces(
        self, mock_lm, sample_instances, mock_tokenizer
    ):
        """Test _encode_pair moves trailing spaces from context to continuation"""
        mock_lm.tokenizer = mock_tokenizer
        for instance in sample_instances:
            context, cont = instance.args
            context_enc, cont_enc = mock_lm._encode_pair(context + " ", cont)
            assert context == mock_lm.tokenizer.decode(context_enc)
            assert " " + cont == mock_lm.tokenizer.decode(cont_enc)

    def test_encode_pair_multiple_trailing_spaces(
        self, mock_lm, sample_instances, mock_tokenizer
    ):
        """Test _encode_pair with multiple trailing spaces"""
        mock_lm.tokenizer = mock_tokenizer
        spaces = [random.randint(4, 10) for _ in range(len(sample_instances))]
        for i, instance in zip(spaces, sample_instances):
            context, cont = instance.args
            context_enc, cont_enc = mock_lm._encode_pair(context + " " * i, cont)
            assert context == mock_lm.tokenizer.decode(context_enc)
            assert " " * i + cont == mock_lm.tokenizer.decode(cont_enc)

    @patch("transformers.AutoModelForSeq2SeqLM")
    def test_encode_pair_seq2seq_model(self, mock_seq2seq, mock_lm):
        """Test _encode_pair behavior with Seq2Seq models"""
        # TODO: Implement test
        pass

    def test_encode_pair_decoder_only_model(self, mock_lm):
        """Test _encode_pair behavior with decoder-only models"""
        # TODO: Implement test
        pass

    # ------------------------------------------------------------------------
    # loglikelihood Tests
    # ------------------------------------------------------------------------

    def test_loglikelihood_add_special_adds_bos(
        self, mock_lm, mock_tokenizer, sample_instances
    ):
        """Test loglikelihood with simple requests"""
        """Testing edge case where context is empty and
        add_special_tokens=True -> encode(context + cont) -> cont == [0] + ..."""
        mock_lm.tokenizer = mock_tokenizer

    def test_loglikelihood_disable_tqdm(self, mock_lm, sample_instances):
        """Test loglikelihood with disable_tqdm=True"""
        # TODO: Implement test
        pass

    def test_loglikelihood_calls_loglikelihood_tokens(self, mock_lm, sample_instances):
        """Test that loglikelihood properly calls _loglikelihood_tokens"""
        # TODO: Implement test
        # Mock _loglikelihood_tokens and verify it's called with correct args
        pass

    # ------------------------------------------------------------------------
    # chat_template Tests
    # ------------------------------------------------------------------------

    def test_chat_template_no_tokenizer(self, mock_lm):
        """Test chat_template returns empty string when tokenizer is None"""
        # TODO: Implement test
        pass

    def test_chat_template_false_returns_none(self, mock_lm_with_tokenizer):
        """Test chat_template returns None when passed False"""
        # TODO: Implement test
        pass

    def test_chat_template_none_returns_none(self, mock_lm_with_tokenizer):
        """Test chat_template returns None when passed None"""
        # TODO: Implement test
        pass

    def test_chat_template_single_template(self, mock_lm_with_tokenizer):
        """Test chat_template with single template string"""
        # TODO: Implement test
        pass

    def test_chat_template_dict_with_default(self, mock_lm_with_tokenizer):
        """Test chat_template with dict containing 'default' key"""
        # TODO: Implement test
        pass

    def test_chat_template_dict_with_specific_name(self, mock_lm_with_tokenizer):
        """Test chat_template with dict and specific template name"""
        # TODO: Implement test
        pass

    def test_chat_template_dict_no_default_raises_error(self, mock_lm_with_tokenizer):
        """Test chat_template raises error when dict has no default"""
        # TODO: Implement test
        pass

    def test_chat_template_dict_invalid_name_raises_error(self, mock_lm_with_tokenizer):
        """Test chat_template raises error for invalid template name"""
        # TODO: Implement test
        pass

    def test_chat_template_uses_default_template(self, mock_lm_with_tokenizer):
        """Test chat_template falls back to default_chat_template"""
        # TODO: Implement test
        pass

    def test_chat_template_warning_for_default(self, mock_lm_with_tokenizer):
        """Test that using default template generates warning"""
        # TODO: Implement test
        pass

    # ------------------------------------------------------------------------
    # Integration Tests
    # ------------------------------------------------------------------------

    def test_loglikelihood_encode_pair_integration(self, mock_lm):
        """Integration test: loglikelihood properly uses _encode_pair"""
        # TODO: Implement test
        pass

    def test_tokenization_consistency(self, mock_lm):
        """Test that tokenization is consistent across multiple calls"""
        # TODO: Implement test
        pass


# ============================================================================
# Additional Helper Functions
# ============================================================================


def create_mock_instance(
    context: str, continuation: str, request_type: str = "loglikelihood"
) -> Instance:
    """
    Helper function to create mock Instance objects for testing.

    Args:
        context: Context string
        continuation: Continuation string
        request_type: Type of request (default: "loglikelihood")

    Returns:
        Instance object with the specified parameters
    """
    return Instance(
        request_type=request_type,
        doc={"context": context, "continuation": continuation},
        arguments=(context, continuation),
        idx=0,
    )


def create_mock_tokenizer_with_chat_templates(
    templates: dict | str | None = None,
) -> Mock:
    """
    Helper function to create a mock tokenizer with specific chat templates.

    Args:
        templates: Chat template(s) - can be None, str, or dict

    Returns:
        Mock tokenizer object with chat_template set
    """
    tokenizer = Mock()
    if isinstance(templates, dict):
        tokenizer.chat_template = templates
        tokenizer.default_chat_template = "default"
    elif isinstance(templates, str):
        tokenizer.chat_template = templates
        tokenizer.default_chat_template = None
    else:
        tokenizer.chat_template = None
        tokenizer.default_chat_template = "default"

    return tokenizer


if __name__ == "__main__":
    pytest.main()
