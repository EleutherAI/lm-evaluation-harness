import pytest

from lm_eval.api.metrics import f1_score, matthews_corrcoef


@pytest.mark.parametrize("aggregation", [f1_score, matthews_corrcoef])
def test_empty_items_raise_the_metric_library_message(aggregation):
    with pytest.raises(ValueError, match="empty"):
        aggregation([])


def test_scores_are_unchanged_for_real_items():
    items = [(1, 1), (0, 0), (1, 0)]

    assert round(f1_score(items), 4) == round(2 / 3, 4)
    assert matthews_corrcoef(items) == 0.0
