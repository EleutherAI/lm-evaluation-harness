"""
Test for issue #3355: AssertionError when parse_generations returns None values
https://github.com/EleutherAI/lm-evaluation-harness/issues/3355
"""

from unittest.mock import patch

from lm_eval.api.instance import Instance
from lm_eval.models.openai_completions import LocalCompletionsAPI


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
            # With the fix, this should complete successfully even with None values
            # The fix converts None to "" (empty string) to maintain the correct count
            results = api.generate_until(requests, disable_tqdm=True)

            # Verify we got all 3 results back
            assert len(results) == 3
            # The second result should be empty string (converted from None)
            assert results[0] == "Response 1"
            assert results[1] == ""  # None was converted to empty string
            assert results[2] == "Response 3"


def test_loglikelihood_with_none_values_from_parse_logprobs():
    """
    Test that verifies the fix for loglikelihood where parse_logprobs
    returns None values. The fix should convert None to a default tuple.
    """
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend="huggingface",
        model="EleutherAI/pythia-1b",
        batch_size=3,  # Process all 3 at once
    )

    # Create mock requests for loglikelihood
    # Note: arguments for loglikelihood is ((context, continuation), additional_kwargs)
    requests = [
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Context 1", "Continuation 1"),
            idx=0,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Context 2", "Continuation 2"),
            idx=1,
        ),
        Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("Context 3", "Continuation 3"),
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
            # With the fix, this should complete successfully even with None values
            # The fix converts None to (-inf, False) to maintain the correct count
            results = api.loglikelihood(requests, disable_tqdm=True)

            # Verify we got all 3 results back
            assert len(results) == 3
            # Verify the values
            assert results[0] == (-1.5, True)
            assert results[1] == (-float("inf"), False)  # None was converted
            assert results[2] == (-2.0, True)


def test_generate_until_sequential_batch_with_none():
    """
    More realistic test simulating what happens in the issue report:
    Multiple batches are processed, and one batch has None values.
    Verifies the fix handles this correctly.
    """
    api = LocalCompletionsAPI(
        base_url="http://test-url.com",
        tokenizer_backend=None,
        model="test-model",
        batch_size=2,  # Process 2 at a time
    )

    # Create 5 requests to be processed in batches of 2
    requests = [
        Instance(
            request_type="generate_until",
            doc={},
            arguments=(f"Context {i}", {"max_gen_toks": 10}),
            idx=i,
        )
        for i in range(5)
    ]

    # Mock parse_generations to inject None on second batch
    original_parse_generations = api.parse_generations
    call_count = [0]

    def mock_parse_generations(outputs, **kwargs):
        call_count[0] += 1
        result = original_parse_generations(outputs, **kwargs)
        # On the second call (second batch), inject a None value
        if call_count[0] == 2 and len(result) >= 2:
            result[1] = None
        return result

    def mock_model_call(*args, **kwargs):
        # Return normal responses
        return {
            "choices": [
                {"index": 0, "text": f"Response batch{call_count[0]}-1"},
                {"index": 1, "text": f"Response batch{call_count[0]}-2"},
            ]
        }

    with patch.object(api, "model_call", side_effect=mock_model_call):
        with patch.object(api, "parse_generations", side_effect=mock_parse_generations):
            # With the fix, this should complete successfully
            results = api.generate_until(requests, disable_tqdm=True)

            # Verify we got all 5 results back
            assert len(results) == 5
            # The 4th result (index 3, second item of second batch) should be empty string
            assert results[3] == ""  # This was None, converted to empty string
