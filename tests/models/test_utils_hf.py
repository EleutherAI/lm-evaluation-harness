from unittest.mock import MagicMock

from lm_eval.models.utils_hf import stop_sequences_criteria


class TestStopSequencesCriteria:
    def test_empty_stop_sequence_builds_no_criteria(self):
        # "" is a substring of every string, so a criterion built from it reports done
        # after the first generated token and truncates every generation to one token.
        tokenizer = MagicMock()
        assert len(stop_sequences_criteria(tokenizer, [""], 0, 1)) == 0

    def test_non_empty_stop_sequence_is_kept(self):
        tokenizer = MagicMock()
        criteria = stop_sequences_criteria(tokenizer, ["\n\n"], 0, 1)
        assert len(criteria) == 1
        assert criteria[0].sequence == "\n\n"

    def test_empty_entries_are_dropped_and_the_rest_preserved(self):
        tokenizer = MagicMock()
        criteria = stop_sequences_criteria(tokenizer, ["</s>", "", "Q:"], 0, 1)
        assert [c.sequence for c in criteria] == ["</s>", "Q:"]

    def test_no_built_criterion_matches_every_token(self):
        # postprocess_generated_text already ignores "" in the stop list, so the two
        # halves of the pipeline should agree on what an empty stop sequence means.
        tokenizer = MagicMock()
        for stop in ([""], ["</s>", ""], ["", "Q:"]):
            assert all(
                c.sequence for c in stop_sequences_criteria(tokenizer, stop, 0, 1)
            )
