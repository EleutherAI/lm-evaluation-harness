from lm_eval.api.metrics import mean
from lm_eval.evaluator_utils import _collect_results
from tests.test_evaluator_utils import MockEvalTask, make_result_acc


def test_metadata_num_fewshot_overrides_zero_config():
    task = MockEvalTask(
        "hardcoded_fewshot",
        config_dict={"num_fewshot": 0, "metadata": {"num_fewshot": 5}},
        agg={"acc": mean},
    )
    acc = make_result_acc(task, {("acc", "none"): [1.0]})

    result = _collect_results({"hardcoded_fewshot": acc}, bootstrap_iters=0)

    assert result.num_fewshot["hardcoded_fewshot"] == 5


def test_metadata_num_fewshot_does_not_override_sampled_examples():
    task = MockEvalTask(
        "sampled_fewshot",
        config_dict={"num_fewshot": 3, "metadata": {"num_fewshot": 5}},
        agg={"acc": mean},
    )
    acc = make_result_acc(task, {("acc", "none"): [1.0]})

    result = _collect_results({"sampled_fewshot": acc}, bootstrap_iters=0)

    assert result.num_fewshot["sampled_fewshot"] == 3
