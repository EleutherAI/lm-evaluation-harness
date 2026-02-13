import unittest.mock as mock

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import _bootstrap_internal_no_mp, mean
from lm_eval.config.utils import parse_metric
from lm_eval.scorers import build_scorers_from_config


def _make_mc_instances(resps, gold, choices, doc_id=0):
    """Helper to create MC instances with pre-set resps and scoring_context."""
    scoring_ctx = {
        "choices": choices,
        "multiple_input": False,
        "multiple_target": False,
    }
    instances = []
    for i, resp in enumerate(resps):
        inst = Instance(
            request_type="loglikelihood",
            doc={},
            arguments=("ctx", "cont"),
            idx=i,
            task_name="test",
            doc_id=doc_id,
            target=gold,
            scoring_context=scoring_ctx,
        )
        inst.resps.append(resp)
        instances.append(inst)
    return instances


def _build_mc_scorer(metric_names):
    """Build a scorer for multiple_choice with given metrics."""
    metrics = [parse_metric({"metric": m}) for m in metric_names]
    scorers = build_scorers_from_config(None, metrics, output_type="multiple_choice")
    return scorers[0]


def test_acc_mutual_info_slicing():
    """Test that acc_mutual_info correctly slices conditional and unconditional loglikelihoods"""

    scorer = _build_mc_scorer(["acc", "acc_mutual_info"])

    # Simulate loglikelihood results for 3 choices
    # Conditional: [-2.0, -1.0, -3.0] - Choice B (index 1) has highest prob
    # Unconditional: [-2.5, -2.0, -2.5]
    resps = [
        (-2.0, False),
        (-1.0, True),
        (-3.0, False),  # Conditional
        (-2.5, False),
        (-2.0, False),
        (-2.5, False),  # Unconditional
    ]

    gold = 1  # Choice "B" is correct
    choices = ["A", "B", "C"]
    instances = _make_mc_instances(resps, gold, choices)

    results = scorer.score(instances)

    # Extract metric values by name
    result_dict = {k[0]: v for k, v in results.items()}

    assert "acc" in result_dict
    assert "acc_mutual_info" in result_dict

    assert result_dict["acc"] == 1.0, f"Expected acc=1.0, got {result_dict['acc']}"
    assert result_dict["acc_mutual_info"] == 1.0, (
        f"Expected acc_mutual_info=1.0, got {result_dict['acc_mutual_info']}"
    )


def test_acc_mutual_info_different_predictions():
    """Test case where conditional and mutual info predictions differ"""

    scorer = _build_mc_scorer(["acc", "acc_mutual_info"])

    # Mutual info calculation:
    # Conditional:   A=-1.0, B=-2.0, C=-3.0 (A wins conditionally)
    # Unconditional: A=-0.5, B=-2.0, C=-3.0
    # Mutual info = conditional - unconditional:
    # A: -1.0 - (-0.5) = -0.5
    # B: -2.0 - (-2.0) = 0.0    <- B wins with mutual info!
    # C: -3.0 - (-3.0) = 0.0
    resps = [
        (-1.0, True),
        (-2.0, False),
        (-3.0, False),  # Conditional (A wins)
        (-0.5, False),
        (-2.0, False),
        (-3.0, False),  # Unconditional
    ]

    gold = 1  # Choice "B" is correct
    choices = ["A", "B", "C"]
    instances = _make_mc_instances(resps, gold, choices)

    results = scorer.score(instances)
    result_dict = {k[0]: v for k, v in results.items()}

    # Regular acc should be 0.0 (A predicted, but B is correct)
    assert result_dict["acc"] == 0.0, f"Expected acc=0.0, got {result_dict['acc']}"

    # Mutual info should be 1.0 (B predicted with mutual info, and B is correct)
    assert result_dict["acc_mutual_info"] == 1.0, (
        f"Expected acc_mutual_info=1.0, got {result_dict['acc_mutual_info']}"
    )


def test_acc_mutual_info_without_metric():
    """Test that normal behavior works when acc_mutual_info is not in metric list"""

    scorer = _build_mc_scorer(["acc"])

    # Only conditional loglikelihoods (no unconditional since acc_mutual_info not requested)
    resps = [(-2.0, False), (-1.0, True), (-3.0, False)]  # 3 choices, B wins

    gold = 1
    choices = ["A", "B", "C"]
    instances = _make_mc_instances(resps, gold, choices)

    results = scorer.score(instances)
    result_dict = {k[0]: v for k, v in results.items()}

    # Should only have acc, not acc_mutual_info
    assert "acc" in result_dict
    assert "acc_mutual_info" not in result_dict
    assert result_dict["acc"] == 1.0


def test_bootstrap_internal_no_mp():
    """Test basic functionality of _bootstrap_internal_no_mp"""

    data = [1, 2, 3, 4, 5]

    # Mock tqdm to avoid progress bar output during testing
    with mock.patch("tqdm.tqdm") as mock_tqdm:
        mock_tqdm.return_value = range(1)  # Single chunk

        # Mock print to avoid output during testing
        with mock.patch("builtins.print"):
            result = _bootstrap_internal_no_mp(mean, data, 100)

    # Should return 100 bootstrap replicates
    assert len(result) == 100

    # All results should be numbers (means)
    assert all(isinstance(x, (int, float)) for x in result)

    # Bootstrap means should be close to original mean
    bootstrap_mean = mean(result)
    original_mean = mean(data)
    assert abs(bootstrap_mean - original_mean) < 0.5  # Should be reasonably close


if __name__ == "__main__":
    test_acc_mutual_info_slicing()
    test_acc_mutual_info_different_predictions()
    test_acc_mutual_info_without_metric()
    test_bootstrap_internal_no_mp()
    print("All tests passed!")
