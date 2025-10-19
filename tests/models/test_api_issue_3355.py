"""
Test for issue #3355: AssertionError when parse_generations returns None values
https://github.com/EleutherAI/lm-evaluation-harness/issues/3355
"""
import pytest
from unittest.mock import patch, MagicMock

from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI
from lm_eval.api.instance import Instance


def test_generate_until_with_none_values_from_parse_generations():
    """
    Test that reproduces issue #3355 where parse_generations returns None values
    causing AssertionError in Collator.get_original() due to mismatched counts.

    This simulates the scenario where the API returns responses with gaps in
    choice indices, leading to None values in the parsed results.
    """
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        batch_size=3,  # Process all 3 requests in one batch
    )

    # Create mock requests
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("Context 1", {"max_gen_toks": 10}),
            idx=0,
        ),
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("Context 2", {"max_gen_toks": 10}),
            idx=1,
        ),
        Instance(
            request_type="generate_until",
            doc={},
            arguments=("Context 3", {"max_gen_toks": 10}),
            idx=2,
        ),
    ]

    # Mock parse_generations directly to return None values
    # This simulates the real scenario where API responses have issues
    original_parse_generations = api.parse_generations

    def mock_parse_generations(outputs, **kwargs):
        # Call original but inject None values
        result = original_parse_generations(outputs, **kwargs)
        # Replace some results with None to simulate the issue
        if len(result) >= 3:
            result[1] = None  # Make the second result None
        return result

    def mock_model_call(*args, **kwargs):
        return {
            "choices": [
                {"index": 0, "text": "Response 1"},
                {"index": 1, "text": "Response 2"},
                {"index": 2, "text": "Response 3"},
            ]
        }

    with patch.object(api, "model_call", side_effect=mock_model_call):
        with patch.object(api, "parse_generations", side_effect=mock_parse_generations):
            # This should raise AssertionError with the original code
            # because parse_generations returns ["Response 1", None, "Response 3"]
            # but the original code skips None values with "if generated_text is not None"
            # resulting in 2 items instead of 3, causing Collator.get_original() to fail
            with pytest.raises(AssertionError):
                results = api.generate_until(requests, disable_tqdm=True)


def test_loglikelihood_with_none_values_from_parse_logprobs():
    """
    Test that reproduces similar issue for loglikelihood where parse_logprobs
    returns None values causing AssertionError in Collator.get_original().
    """
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend="huggingface",
        model="EleutherAI/pythia-1b"
    )

    # Create mock requests for loglikelihood
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(("Context 1", "Continuation 1"), {}),
            idx=0,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(("Context 2", "Continuation 2"), {}),
            idx=1,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=(("Context 3", "Continuation 3"), {}),
            idx=2,
        ),
    ]

    # Mock parse_logprobs to return a list with None values
    def mock_parse_logprobs(*args, **kwargs):
        # Simulate parse_logprobs returning None for some values
        return [(-1.5, True), None, (-2.0, True)]

    def mock_model_call(*args, **kwargs):
        return {"choices": [{"index": 0}]}

    with patch.object(api, "model_call", side_effect=mock_model_call):
        with patch.object(api, "parse_logprobs", side_effect=mock_parse_logprobs):
            # This should raise AssertionError with the original code
            # because the code skips None values when appending to res
            with pytest.raises(AssertionError):
                results = api.loglikelihood(requests, disable_tqdm=True)


def test_generate_until_sequential_batch_with_none():
    """
    More realistic test simulating what happens in the issue report:
    Multiple batches are processed, and one batch has None values.
    """
    api = LocalChatCompletion(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        batch_size=2,  # Process 2 at a time
    )

    # Create 5 requests
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(f"Context {i}", {"max_gen_toks": 10}),
            idx=i,
        )
        for i in range(5)
    ]

    call_count = [0]

    def mock_model_call(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:
            # Second batch returns a response with a gap, producing None
            return {
                "choices": [
                    {"index": 0, "message": {"content": "Good response"}},
                    {"index": 2, "message": {"content": "Another response"}},  # Gap!
                ]
            }
        else:
            # Other batches are normal
            return {
                "choices": [
                    {"index": 0, "message": {"content": f"Response {call_count[0]}-1"}},
                    {"index": 1, "message": {"content": f"Response {call_count[0]}-2"}},
                ]
            }

    with patch.object(api, "model_call", side_effect=mock_model_call):
        # This should raise AssertionError with the original code
        with pytest.raises(AssertionError):
            results = api.generate_until(requests, disable_tqdm=True)
