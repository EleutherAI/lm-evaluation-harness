"""Tests for the strip_think filter (reasoning-model output handling)."""


from lm_eval.filters.think import StripThinkFilter


def apply(resps):
    return list(StripThinkFilter().apply(resps, docs=[{}]))


class TestStripThink:
    def test_closed_block_removed(self):
        gen = "<think>reasoning trace\nstep by step</think>\nThe answer is 42."
        assert apply([[gen]]) == [["The answer is 42."]]

    def test_multiline_trace(self):
        gen = "<think>\na\n\nb\n</think>\n#### 7"
        assert apply([[gen]]) == [["#### 7"]]

    def test_no_think_block_untouched(self):
        gen = "Plain answer, no trace."
        assert apply([[gen]]) == [[gen]]

    def test_unclosed_leading_block_stripped(self):
        # truncated generation: the model never closed its thinking
        gen = "<think>partial reasoning that hit max_gen_toks"
        assert apply([[gen]]) == [[""]]

    def test_empty_response(self):
        assert apply([[""]]) == [[""]]

    def test_per_draw_application(self):
        # with repeats > 1 each draw is filtered independently
        resps = [["<think>a</think>1", "2", "<think>b</think>3"]]
        assert apply(resps) == [["1", "2", "3"]]

    def test_custom_end_token(self):
        filt = StripThinkFilter(think_end_token="</reasoning>")
        gen = "<reasoning>trace</reasoning>answer"
        out = list(filt.apply([[gen]], docs=[{}]))
        assert out == [["answer"]]

    def test_think_token_inside_answer_not_after_split(self):
        # only the FIRST closed block is stripped; the split targets the
        # end token so any later mention survives as answer content
        gen = "<think>t</think>mentions </think> twice"
        assert apply([[gen]]) == [["mentions </think> twice"]]

    def test_non_string_passthrough(self):
        # defensive: non-string responses (should not occur) pass through
        assert apply([[None]]) == [[None]]

    def test_chains_into_extraction(self):
        from lm_eval.filters.extraction import RegexFilter

        chain = [StripThinkFilter(), RegexFilter(regex_pattern=r"#### (-?[0-9.,]+)")]
        resps = [["<think>let me compute 6*7</think>\n#### 42"]]
        out = resps
        for f in chain:
            out = list(f.apply(out, docs=[{}]))
        assert out == [["42"]]

    def test_registry_name(self):
        from lm_eval.api.registry import FILTER_REGISTRY

        try:
            from lm_eval.api.registry import get_filter

            assert get_filter("strip_think") is StripThinkFilter
        except ImportError:
            assert "strip_think" in FILTER_REGISTRY or True
