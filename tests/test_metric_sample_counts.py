from lm_eval.api.metrics import mean
from lm_eval.evaluator_utils import _compute_task_aggregations


class _Task:
    task_name = "test_task"

    def aggregation(self):
        return {"acc": mean, "f1": mean}


def test_task_aggregation_reports_each_metric_count():
    metrics, sample_len = _compute_task_aggregations(
        _Task(),
        {
            ("acc", "none"): [1.0],
            ("f1", "none"): [0.5] * 100,
        },
        bootstrap_iters=0,
    )

    assert sample_len == 100
    assert metrics["sample_count"] == {"acc,none": 1, "f1,none": 100}
