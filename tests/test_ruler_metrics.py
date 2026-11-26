from lm_eval.tasks.ruler.common_utils import string_match_all, string_match_part


def test_part_gives_full_credit_for_any_matching_reference():
    assert (
        string_match_part(["the answer is paris"], [["Paris", "London", "Rome"]]) == 1.0
    )


def test_part_no_match():
    assert string_match_part(["berlin"], [["Paris", "London"]]) == 0.0


def test_part_averages_over_predictions():
    preds = ["paris", "berlin"]
    refs = [["Paris", "City of Light"], ["Rome"]]
    assert string_match_part(preds, refs) == 0.5


def test_all_still_requires_every_reference():
    assert string_match_all(["paris"], [["Paris", "London"]]) == 0.5
