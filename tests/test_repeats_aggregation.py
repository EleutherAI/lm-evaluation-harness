import logging

import pytest

from lm_eval.api.metrics import pass_at_k_fn
from lm_eval.filters.extraction import RegexFilter
from lm_eval.filters.selection import MajorityVoteFilter, TakeFirstFilter, TakeKFilter


class TestPassAtKMetric:
    def test_any_correct(self):
        items = [(["42", "41", "42", "42"], "42")]
        assert pass_at_k_fn(items) == [1.0]

    def test_one_of_k_correct(self):
        items = [(["41", "41", "42", "41"], "42")]
        assert pass_at_k_fn(items) == [1.0]

    def test_none_correct(self):
        items = [(["41", "40", "43", "44"], "42")]
        assert pass_at_k_fn(items) == [0.0]

    def test_k1_degenerates_to_exact_match(self):
        items = [(["42"], "42"), (["41"], "42")]
        assert pass_at_k_fn(items) == [1.0, 0.0]

    def test_single_string_response(self):
        # the default take_first path yields a bare string response
        items = [("42", "42"), ("41", "42")]
        assert pass_at_k_fn(items) == [1.0, 0.0]

    def test_mixed_documents(self):
        items = [
            (["1", "2", "1", "1"], "1"),
            (["2", "2", "2", "2"], "1"),
            (["3", "1"], "1"),
        ]
        assert pass_at_k_fn(items) == [1.0, 0.0, 1.0]

    def test_non_string_gold_coerced(self):
        items = [(["42"], 42)]
        assert pass_at_k_fn(items) == [1.0]


class TestMajorityVoteFilter:
    def test_modal_response_selected(self):
        filt = MajorityVoteFilter()
        out = list(filt.apply([["41", "42", "42", "43"]], docs=[{}]))
        assert out == [["42"]]

    def test_tie_breaks_to_first_encountered(self):
        # Counter.most_common breaks ties by insertion order: the first
        # response to reach the tied count wins.
        filt = MajorityVoteFilter()
        out = list(filt.apply([["41", "42", "41", "42"]], docs=[{}]))
        assert out == [["41"]]

    def test_unanimous(self):
        filt = MajorityVoteFilter()
        out = list(filt.apply([["42", "42", "42"]], docs=[{}]))
        assert out == [["42"]]

    def test_single_response(self):
        filt = MajorityVoteFilter()
        out = list(filt.apply([["42"]], docs=[{}]))
        assert out == [["42"]]


class TestTakeKFilter:
    def test_keeps_first_k(self):
        filt = TakeKFilter(k=4)
        out = list(filt.apply([["a", "b", "c", "d", "e"]], docs=[{}]))
        assert out == [["a", "b", "c", "d"]]

    def test_insufficient_responses_raises(self):
        filt = TakeKFilter(k=8)
        with pytest.raises(AssertionError, match="increase TaskConfig.repeats"):
            list(filt.apply([["a", "b"]], docs=[{}]))


class TestExtractionThenMajorityPipeline:
    def test_chained_per_draw(self):
        # the self-consency recipe: extract each draw, then vote
        extraction = RegexFilter(
            regex_pattern=r"#### (-?[\d,\.]+)", fallback="[invalid]"
        )
        vote = MajorityVoteFilter()
        resps = [
            ["reasoning #### 42", "reasoning #### 41", "reasoning #### 42"],
        ]
        extracted = list(extraction.apply(resps, docs=[{}]))
        assert extracted == [["42", "41", "42"]]
        voted = list(vote.apply(extracted, docs=[{}]))
        assert voted == [["42"]]


class TestTakeFirstRegression:
    def test_default_unaffected(self):
        filt = TakeFirstFilter()
        out = list(filt.apply([["a", "b", "c"]], docs=[{}]))
        assert out == ["a"]


class TestRepeatsConfig:
    def test_taskconfig_repeats_default(self):
        from lm_eval.config.task import TaskConfig

        assert TaskConfig().repeats == 1

    def test_task_config_repeats_field(self):
        from lm_eval.config.task import TaskConfig

        cfg = TaskConfig(repeats=8)
        assert cfg.repeats == 8
        cfg.repeats = 4
        assert cfg.repeats == 4

    def test_simple_evaluate_exposes_repeats(self):
        # the Python-API entrypoint must accept repeats (issue #3339)
        import inspect

        from lm_eval.evaluator import simple_evaluate

        assert "repeats" in inspect.signature(simple_evaluate).parameters


def _tiny_task_config(repeats: int):
    from lm_eval.config.task import TaskConfig

    return TaskConfig(
        task=f"unit_test_repeats_{repeats}",
        dataset_path="",
        output_type="generate_until",
        doc_to_text="question",
        doc_to_target="answer",
        generation_kwargs={"do_sample": True, "temperature": 0.7},
        repeats=repeats,
        test_split="test",
    )


def _build_task(repeats: int, monkeypatch):
    from lm_eval.api.task import ConfigurableTask

    def fake_download(self, dataset_kwargs=None, **kwargs):
        import datasets

        self.dataset = {
            "test": datasets.Dataset.from_list(
                [
                    {"question": "2+2?", "answer": "4"},
                    {"question": "3+3?", "answer": "6"},
                ]
            )
        }

    monkeypatch.setattr(ConfigurableTask, "download", fake_download)
    return ConfigurableTask(config=_tiny_task_config(repeats))


class TestDiscardWarning:
    def test_warning_fires_when_repeats_discarded(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING):
            _build_task(repeats=4, monkeypatch=monkeypatch)
        assert "take_first" in caplog.text
        assert "majority_vote" in caplog.text
        assert "pass_at_k" in caplog.text

    def test_no_warning_at_repeats_one(self, caplog, monkeypatch):
        with caplog.at_level(logging.WARNING):
            task = _build_task(repeats=1, monkeypatch=monkeypatch)
        assert "take_first" not in caplog.text
        # the default single-draw pipeline is unchanged
        assert len(task._filters) == 1


class TestConfigRoundtripPreservesFields:
    """Regression for the silent field loss (issue #3339): a TaskConfig
    passed to ConfigurableTask was unpacked as **kwargs, reading only its
    dict items and resetting every dataclass field to its default."""

    def test_taskconfig_object_preserved(self, monkeypatch):
        task = _build_task(repeats=4, monkeypatch=monkeypatch)
        assert task.config.repeats == 4
        assert task.config.task == "unit_test_repeats_4"

    def test_dict_config_still_works(self, monkeypatch):
        from lm_eval.api.task import ConfigurableTask

        def fake_download(self, dataset_kwargs=None, **kwargs):
            import datasets

            self.dataset = {
                "test": datasets.Dataset.from_list(
                    [{"question": "2+2?", "answer": "4"}]
                )
            }

        monkeypatch.setattr(ConfigurableTask, "download", fake_download)
        cfg = {
            "task": "unit_test_dict_cfg",
            "dataset_path": "",
            "output_type": "generate_until",
            "doc_to_text": "question",
            "doc_to_target": "answer",
            "repeats": 4,
            "test_split": "test",
        }
        task = ConfigurableTask(config=cfg)
        assert task.config.repeats == 4
