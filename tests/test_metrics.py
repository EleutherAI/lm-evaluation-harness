import unittest.mock as mock

from lm_eval.api.metrics import _bootstrap_internal_no_mp, mean
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig


class MockConfigurableTask(ConfigurableTask):
    """Mock task for testing metrics"""

    def __init__(self):
        # Create a minimal config
        config = {
            "task": "test_acc_mutual_info",
            "output_type": "multiple_choice",
            "metric_list": [{"metric": "acc"}, {"metric": "acc_mutual_info"}],
            "doc_to_choice": ["A", "B", "C"],
            "doc_to_target": 1,  # Correct answer is index 1 (choice "B")
            "target_delimiter": " ",
        }

        # Initialize with minimal setup
        self._config = TaskConfig(**config)
        self.OUTPUT_TYPE = "multiple_choice"

        # Set up required attributes
        self.multiple_input = 0
        self.multiple_target = 0

        # Set up metrics
        self._metric_fn_list = {"acc": None, "acc_mutual_info": None}
        self._metric_fn_kwargs = {"acc": {}, "acc_mutual_info": {}}
        self._aggregation_list = {}
        self._higher_is_better = {}

    def doc_to_choice(self, doc):
        return ["A", "B", "C"]

    def doc_to_target(self, doc):
        return 1  # Choice "B" is correct

    # Required abstract methods (minimal implementations)
    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def download(self, **kwargs):
        pass


def test_acc_mutual_info_slicing():
    """Test that acc_mutual_info correctly slices conditional and unconditional loglikelihoods"""

    task = MockConfigurableTask()

    # Simulate loglikelihood results for 3 choices
    # Format: [(loglikelihood, is_greedy), ...]
    # First 3 are conditional P(choice|context), next 3 are unconditional P(choice)

    # Combined results as they would come from the model
    # Order: conditional_1, conditional_2, conditional_3, unconditional_1, unconditional_2, unconditional_3
    # Conditional: [-2.0, -1.0, -3.0] - Choice B (index 1) has highest prob
    # Unconditional: [-2.5, -2.0, -2.5] - Choice B has higher unconditional prob too
    results = [
        (-2.0, False),
        (-1.0, True),
        (-3.0, False),  # Conditional
        (-2.5, False),
        (-2.0, False),
        (-2.5, False),
    ]  # Unconditional

    # Test the process_results method
    doc = {}  # Mock document
    result_dict = task.process_results(doc, results)

    # Verify that both acc and acc_mutual_info are calculated
    assert "acc" in result_dict
    assert "acc_mutual_info" in result_dict

    # Both should be 1.0 since choice B (index 1) is correct and has highest probability
    assert result_dict["acc"] == 1.0, f"Expected acc=1.0, got {result_dict['acc']}"
    assert result_dict["acc_mutual_info"] == 1.0, (
        f"Expected acc_mutual_info=1.0, got {result_dict['acc_mutual_info']}"
    )


def test_acc_mutual_info_different_predictions():
    """Test case where conditional and mutual info predictions differ"""

    task = MockConfigurableTask()

    # Mutual info calculation:
    # Conditional:   A=-1.0, B=-2.0, C=-3.0 (A wins conditionally)
    # Unconditional: A=-0.5, B=-2.0, C=-3.0 (A has much higher unconditional prob)
    # Mutual info = conditional - unconditional:
    # A: -1.0 - (-0.5) = -0.5
    # B: -2.0 - (-2.0) = 0.0    <- B wins with mutual info!
    # C: -3.0 - (-3.0) = 0.0

    results = [
        (-1.0, True),
        (-2.0, False),
        (-3.0, False),  # Conditional (A wins)
        (-0.5, False),
        (-2.0, False),
        (-3.0, False),
    ]  # Unconditional

    doc = {}
    result_dict = task.process_results(doc, results)

    # Regular acc should be 0.0 (A predicted, but B is correct)
    assert result_dict["acc"] == 0.0, f"Expected acc=0.0, got {result_dict['acc']}"

    # Mutual info should be 1.0 (B predicted with mutual info, and B is correct)
    assert result_dict["acc_mutual_info"] == 1.0, (
        f"Expected acc_mutual_info=1.0, got {result_dict['acc_mutual_info']}"
    )


def test_acc_mutual_info_without_metric():
    """Test that normal behavior works when acc_mutual_info is not in metric list"""

    # Create task without acc_mutual_info
    config = {
        "task": "test_normal",
        "output_type": "multiple_choice",
        "metric_list": [{"metric": "acc"}],  # Only acc, no acc_mutual_info
        "doc_to_choice": ["A", "B", "C"],
        "doc_to_target": 1,
        "target_delimiter": " ",
    }

    task = MockConfigurableTask()
    task._config = TaskConfig(**config)
    task._metric_fn_list = {"acc": None}  # Only acc

    # Only conditional loglikelihoods (no unconditional since acc_mutual_info not requested)
    results = [(-2.0, False), (-1.0, True), (-3.0, False)]  # 3 choices, B wins

    doc = {}
    result_dict = task.process_results(doc, results)

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
