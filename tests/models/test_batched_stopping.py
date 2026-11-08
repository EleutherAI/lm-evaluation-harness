"""A batch finishes when each row has encountered any configured stop string."""

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from lm_eval.models.utils_hf import MultiTokenEOSCriteria, stop_sequences_criteria


@pytest.fixture
def tokenizer():
    unknown_marker, end_marker = "[UNK]", "[EOS]"
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {
                "prompt": 0,
                "A": 1,
                "B": 2,
                "C": 3,
                "noise": 4,
                unknown_marker: 5,
                end_marker: 6,
            },
            unk_token=unknown_marker,
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=unknown_marker,
        eos_token=end_marker,
        pad_token=end_marker,
    )


def batch_is_done(criteria, rows):
    # Transformers has supported both scalar and per-row stopping results.
    return bool(torch.as_tensor(criteria(torch.tensor(rows), None)).all())


@pytest.mark.parametrize("stops", [["A", "B"], ["B", "A"]])
def test_different_stop_strings_finish_the_same_batch(tokenizer, stops):
    criteria = stop_sequences_criteria(tokenizer, stops, 1, 2)
    assert batch_is_done(criteria, [[0, 1], [0, 2]])


def test_stop_matches_persist_across_generation_steps(tokenizer):
    criteria = stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2)
    assert not batch_is_done(criteria, [[0, 1], [0, 4]])
    # The first row's match has moved outside the current lookback window.
    assert batch_is_done(criteria, [[0, 1, 4, 4, 4, 4], [0, 4, 4, 4, 4, 2]])


@pytest.mark.parametrize("stops", [["A B", "C"], ["C", "A B"]])
def test_different_length_stop_strings_share_completion(tokenizer, stops):
    criteria = stop_sequences_criteria(tokenizer, stops, 1, 2)
    assert not batch_is_done(criteria, [[0, 1], [0, 3]])
    assert batch_is_done(criteria, [[0, 1, 2], [0, 3, 4]])


def test_unfinished_row_keeps_batch_running(tokenizer):
    criteria = stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2)
    assert not batch_is_done(criteria, [[0, 1, 2], [0, 4, 4]])


def test_prompt_stop_strings_are_ignored(tokenizer):
    criteria = stop_sequences_criteria(tokenizer, ["A", "B"], 2, 2)
    assert not batch_is_done(criteria, [[0, 1, 4], [0, 2, 4]])


def test_separate_batches_do_not_share_completion(tokenizer):
    finished = stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2)
    fresh = stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2)
    assert batch_is_done(finished, [[0, 1], [0, 1]])
    assert not batch_is_done(fresh, [[0, 4], [0, 4]])


def test_standalone_criteria_retain_independent_state(tokenizer):
    first = MultiTokenEOSCriteria("A", tokenizer, 1, 2)
    second = MultiTokenEOSCriteria("B", tokenizer, 1, 2)
    assert not first(torch.tensor([[0, 1], [0, 4]]), None)
    assert not second(torch.tensor([[0, 4], [0, 2]]), None)


def test_same_stop_string_finishes_batch(tokenizer):
    criteria = stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2)
    assert batch_is_done(criteria, [[0, 2], [0, 2]])


def test_transformers_generation_stops_after_last_row_matches(tokenizer):
    class ForcedTokens(transformers.LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores.fill_(float("-inf"))
            step = input_ids.shape[1] - 1
            scores[0, 1 if step == 0 else 4] = 0
            scores[1, 2 if step == 1 else 4] = 0
            return scores

    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=0,
            eos_token_id=6,
            pad_token_id=6,
        )
    ).eval()
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            result = model.generate(
                torch.tensor([[0], [0]]),
                attention_mask=torch.ones((2, 1), dtype=torch.long),
                max_new_tokens=6,
                do_sample=False,
                logits_processor=transformers.LogitsProcessorList([ForcedTokens()]),
                stopping_criteria=stop_sequences_criteria(tokenizer, ["A", "B"], 1, 2),
            )
    finally:
        torch.set_num_threads(original_threads)
    assert result.tolist() == [[0, 1, 4], [0, 4, 2]]
