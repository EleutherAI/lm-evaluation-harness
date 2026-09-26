"""Tests for the GPQA and BBH reasoning-model task variants.

These variants (issue #2682) prepend a ``strip_think`` filter to the
existing extraction chains and use generation settings that do not
truncate inside reasoning traces. The tests verify: the filter chains
handle R1-format outputs through the real task machinery, and the
yaml configurations are structurally sound.
"""

import pathlib

import yaml


TASKS_ROOT = pathlib.Path(__file__).parent.parent / "lm_eval/tasks"

GPQA_TEMPLATE = TASKS_ROOT / "gpqa/reasoning/_gpqa_reasoning_yaml"
BBH_TEMPLATE = TASKS_ROOT / "bbh/reasoning/_bbh_reasoning_yaml"
GPQA_CONFIGS = sorted((TASKS_ROOT / "gpqa/reasoning").glob("gpqa_*_reasoning.yaml"))
BBH_CONFIGS = sorted((TASKS_ROOT / "bbh/reasoning").glob("bbh_*_reasoning.yaml"))


def _load(path):
    text = path.read_text().replace("!function", "")
    return yaml.safe_load(text)


class TestGpqaReasoningTemplate:
    def test_strip_think_in_both_filters(self):
        cfg = _load(GPQA_TEMPLATE)
        for filt in cfg["filter_list"]:
            functions = [f["function"] for f in filt["filter"]]
            assert "strip_think" in functions, filt["name"]
            assert functions.index("strip_think") < functions.index("take_first")

    def test_generation_kwargs_reasoning_safe(self):
        cfg = _load(GPQA_TEMPLATE)
        gk = cfg["generation_kwargs"]
        assert gk["max_gen_toks"] >= 16384, "R1 traces need room"
        assert "\n\n" not in gk["until"], "paragraph delimiter truncates traces"
        assert "</s>" in gk["until"]

    def test_metrics_preserved_from_base(self):
        cfg = _load(GPQA_TEMPLATE)
        metrics = [m["metric"] for m in cfg["metric_list"]]
        assert metrics == ["exact_match"]

    def test_three_subsets_generated(self):
        assert len(GPQA_CONFIGS) == 3
        names = [c.stem for c in GPQA_CONFIGS]
        assert "gpqa_diamond_reasoning" in names
        assert "gpqa_main_reasoning" in names
        assert "gpqa_extended_reasoning" in names


class TestBbhReasoningTemplate:
    def test_strip_think_before_extraction(self):
        cfg = _load(BBH_TEMPLATE)
        for filt in cfg["filter_list"]:
            functions = [f["function"] for f in filt["filter"]]
            assert functions[0] == "strip_think", "strip must be first"

    def test_generation_kwargs_reasoning_safe(self):
        cfg = _load(BBH_TEMPLATE)
        gk = cfg["generation_kwargs"]
        assert gk["max_gen_toks"] >= 16384
        assert "\n\n" not in gk["until"]
        assert "Q:" not in gk["until"]

    def test_all_27_subtasks_generated(self):
        assert len(BBH_CONFIGS) == 27, f"expected 27, got {len(BBH_CONFIGS)}"

    def test_subtask_includes_template(self):
        for cfg_path in BBH_CONFIGS[:3]:
            cfg = yaml.safe_load(cfg_path.read_text())
            assert cfg["include"] == "_bbh_reasoning_yaml"
            assert cfg["task"].endswith("_reasoning")


class TestR1FormatThroughChains:
    """The end-to-end property these tasks exist to provide: an R1-format
    response scores correctly through the variant's filter chain."""

    def _build_bbh_chain(self):
        from lm_eval.filters.extraction import RegexFilter
        from lm_eval.filters.selection import TakeFirstFilter
        from lm_eval.filters.think import StripThinkFilter

        return [
            StripThinkFilter(),
            RegexFilter(regex_pattern=r"(?<=the answer is )(.*)(?=.)"),
            TakeFirstFilter(),
        ]

    def _build_gpqa_chain(self):
        from lm_eval.filters.extraction import MultiChoiceRegexFilter
        from lm_eval.filters.selection import TakeFirstFilter
        from lm_eval.filters.think import StripThinkFilter

        return [
            StripThinkFilter(),
            MultiChoiceRegexFilter(
                group_select=-1,
                ignore_case=True,
                ignore_punctuation=True,
                regex_pattern=r"(\([A-Z]\))",
            ),
            TakeFirstFilter(),
        ]

    def test_bbh_r1_response_scores(self):
        chain = self._build_bbh_chain()
        trace = (
            "<think>The question asks about boolean precedence. NOT comes first,"
            " so not True is False, then False and False is False..."
            "</think>\n...the answer is False."
        )
        out = [[trace]]
        for f in chain:
            out = list(f.apply(out, docs=[{}]))
        assert out[0] == "False"

    def test_gpqa_r1_response_scores(self):
        chain = self._build_gpqa_chain()
        trace = (
            "<think>Analyze the quantum numbers... conservation of parity"
            " rules out (A) and (C)... between (B) and (D)...</think>"
            "The answer is (B)."
        )
        out = [[trace]]
        docs = [
            {
                "choices": [
                    "(A) quantum state",
                    "(B) energy level",
                    "(C) particle",
                    "(D) wave",
                ]
            }
        ]
        for f in chain:
            out = list(f.apply(out, docs=docs))
        assert out[0] == "(B)"

    def test_trace_without_answer_never_scores(self):
        chain = self._build_bbh_chain()
        trace = "<think>partial reasoning that hit max_gen_toks"
        out = [[trace]]
        for f in chain:
            out = list(f.apply(out, docs=[{}]))
        assert out[0] != "False"
