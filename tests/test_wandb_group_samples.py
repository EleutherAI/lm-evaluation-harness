from copy import deepcopy
from unittest.mock import Mock

import pytest

from lm_eval.loggers.wandb_logger import WandbLogger


def _sample(doc_id):
    return {
        "doc_id": doc_id,
        "target": 1,
        "arguments": [("question", "A"), ("question", "B")],
        "resps": [[(-2.0, False)], [(-1.0, True)]],
        "filtered_resps": [(-2.0, False), (-1.0, True)],
        "acc": 1.0,
    }


def _logger(group_rows=(), aggregated_groups=(), config_group=None):
    configs = {
        name: {
            "output_type": "multiple_choice",
            "metric_list": [{"metric": "acc"}],
            **({"group": config_group} if config_group else {}),
        }
        for name in ("second", "first")
    }
    results = {
        "results": {
            name: {"alias": name, "acc,none": 1.0}
            for name in ("second", *group_rows, "first")
        },
        "configs": configs,
        "group_subtasks": {name: ["second", "first"] for name in group_rows},
    }
    if aggregated_groups:
        results["groups"] = {name: {} for name in aggregated_groups}
    logger = WandbLogger.__new__(WandbLogger)
    logger.post_init(results)
    logger._get_config()
    logger.step = 7
    logger.run = Mock()
    logger._log_samples_as_artifact = Mock()
    return logger


@pytest.mark.parametrize(
    ("group_rows", "aggregated_groups"),
    [
        ((), ()),
        (("aggregate",), ("aggregate",)),
        (("selection",), ()),
        (("selection", "parent"), ()),
        (("selection", "parent"), ("parent",)),
    ],
    ids=["standalone", "aggregated", "selection", "nested", "mixed"],
)
def test_sample_logging_ignores_result_only_group_rows(group_rows, aggregated_groups):
    logger = _logger(group_rows, aggregated_groups)
    samples = {"first": [_sample(0)], "second": [_sample(1)]}
    original_samples = deepcopy(samples)
    original_results = deepcopy(logger.results)

    logger.log_eval_samples(samples)

    assert [list(call.args[0]) for call in logger.run.log.call_args_list] == [
        ["second_eval_results"],
        ["first_eval_results"],
    ]
    for call, name, doc_id in zip(
        logger.run.log.call_args_list, ("second", "first"), (1, 0), strict=True
    ):
        table = call.args[0][f"{name}_eval_results"]
        assert table["id"].tolist() == [doc_id]
        assert table["filtered_predictions"].tolist() == [1]
        assert table["acc"].tolist() == [1.0]
        assert call.kwargs == {"step": 7}
    assert [call.args for call in logger._log_samples_as_artifact.call_args_list] == [
        (samples["second"], "second"),
        (samples["first"], "first"),
    ]
    assert samples == original_samples
    assert logger.results == original_results


@pytest.mark.parametrize("config_group", ["legacy", ["legacy", "also_legacy"]])
def test_result_only_groups_do_not_change_configured_sample_tables(config_group):
    logger = _logger(("selection",), config_group=config_group)
    samples = {"first": [_sample(0)], "second": [_sample(1)]}

    logger.log_eval_samples(samples)

    groups = [config_group] if isinstance(config_group, str) else config_group
    assert len(logger.run.log.call_args_list) == len(groups)
    for call, group in zip(logger.run.log.call_args_list, groups, strict=True):
        table = call.args[0][f"{group}_eval_results"]
        assert table["task"].tolist() == ["second", "first"]
        assert table["group"].tolist() == [group, group]
        assert table["id"].tolist() == [1, 0]
        assert call.kwargs == {"step": 7}
    assert logger._log_samples_as_artifact.call_count == 2 * len(groups)


def test_partial_sample_mapping_logs_only_supplied_tasks():
    logger = _logger(("selection",))
    samples = {"first": [_sample(0)]}

    logger.log_eval_samples(samples)

    logger.run.log.assert_called_once()
    assert list(logger.run.log.call_args.args[0]) == ["first_eval_results"]
    logger._log_samples_as_artifact.assert_called_once_with(samples["first"], "first")


@pytest.mark.parametrize("config_group", [None, "legacy"])
def test_present_empty_samples_are_still_logged(config_group):
    logger = _logger(("selection",), config_group=config_group)

    logger.log_eval_samples({"first": []})

    logger.run.log.assert_called_once()
    name = config_group or "first"
    assert logger.run.log.call_args.args[0][f"{name}_eval_results"].empty
    logger._log_samples_as_artifact.assert_called_once_with([], "first")


@pytest.mark.parametrize("samples", [{}, {"unknown": [_sample(0)]}])
def test_mapping_without_selected_tasks_does_not_log(samples):
    logger = _logger(("selection",))

    logger.log_eval_samples(samples)

    logger.run.log.assert_not_called()
    logger._log_samples_as_artifact.assert_not_called()


def test_aggregated_groups_are_excluded_even_if_present_in_samples():
    logger = _logger(("aggregate",), ("aggregate",))
    samples = {
        "first": [_sample(0)],
        "second": [_sample(1)],
        "aggregate": [_sample(2)],
    }

    logger.log_eval_samples(samples)

    assert logger.run.log.call_count == 2
    assert [
        call.args[1] for call in logger._log_samples_as_artifact.call_args_list
    ] == [
        "second",
        "first",
    ]
