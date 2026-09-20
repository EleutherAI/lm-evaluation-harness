import sys
import types

import pandas as pd

from lm_eval.loggers.wandb_logger import WandbLogger


class StrictFakeTable:
    def __init__(self, columns=None, dataframe=None, allow_mixed_types=False):
        self.columns = columns
        self.dataframe = dataframe
        self.allow_mixed_types = allow_mixed_types

        if dataframe is not None and not allow_mixed_types:
            for column in dataframe.columns:
                types_seen = {
                    type(value)
                    for value in dataframe[column]
                    if value is not None and not pd.isna(value)
                }
                if len(types_seen) > 1:
                    raise TypeError(f"mixed types in column {column}")


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload, step=None):
        self.logged.append((payload, step))


def test_log_eval_samples_allows_mixed_group_table_types(monkeypatch):
    fake_wandb = types.SimpleNamespace(Table=StrictFakeTable)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    logger = WandbLogger.__new__(WandbLogger)
    logger.run = FakeRun()
    logger.step = 7
    logger.task_names = ["generate_task", "multiple_choice_task", "mixed_group"]
    logger.group_names = ["mixed_group"]
    logger.task_configs = {
        "generate_task": {
            "group": "mixed_group",
            "output_type": "generate_until",
            "metric_list": [{"metric": "exact_match"}],
        },
        "multiple_choice_task": {
            "group": "mixed_group",
            "output_type": "multiple_choice",
            "metric_list": [{"metric": "acc"}],
        },
    }
    monkeypatch.setattr(logger, "_log_samples_as_artifact", lambda *args: None)

    samples = {
        "generate_task": [
            {
                "doc_id": 0,
                "target": "answer text",
                "arguments": [("prompt", {})],
                "resps": [["answer text"]],
                "filtered_resps": ["answer text"],
                "exact_match": 1.0,
            }
        ],
        "multiple_choice_task": [
            {
                "doc_id": 1,
                "target": 1,
                "arguments": [("prompt", "A"), ("prompt", "B")],
                "resps": [[(0.1, False)], [(0.9, True)]],
                "filtered_resps": [(0.1, False), (0.9, True)],
                "acc": 1.0,
            }
        ],
    }

    logger.log_eval_samples(samples)

    assert len(logger.run.logged) == 1
    payload, step = logger.run.logged[0]
    table = payload["mixed_group_eval_results"]
    assert step == 7
    assert isinstance(table, StrictFakeTable)
    assert table.allow_mixed_types is True
    assert set(table.dataframe["task"]) == {"generate_task", "multiple_choice_task"}
