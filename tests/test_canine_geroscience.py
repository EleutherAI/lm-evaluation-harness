import pytest

from lm_eval.tasks.canine_geroscience import utils


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ("13.7 years", "13.7 years.", 1.0),
        ("The overall life expectancy was 13.7 years.", "13.7 years.", 0.0),
        ("thirteen point seven", "13.7 years.", 0.0),
    ],
)
def test_exact_match_is_normalised_equality(pred, gold, expected):
    assert utils.process_results({"answer": gold}, [pred])["exact_match"] == expected


def test_token_f1_rewards_partial_overlap():
    f1 = utils.token_f1("The overall life expectancy was 13.7 years.", "13.7 years.")
    assert 0.0 < f1 < 1.0
    assert utils.token_f1("13.7 years", "13.7 years.") == 1.0
    assert utils.token_f1("no overlap here", "13.7 years.") == 0.0


@pytest.mark.parametrize(
    "pred, gold, primary, every",
    [
        ("12.69 years", "12.69 years (95% CI 12.68-12.70).", 1.0, 1.0),
        ("14 years", "12.69 years (95% CI 12.68-12.70).", 0.0, 0.0),
        ("9.8 years", "9.8 years; 2.1 years less than mesocephalic dogs.", 1.0, 0.0),
        (
            "9.8 years, 2.1 years less",
            "9.8 years; 2.1 years less than mesocephalic dogs.",
            1.0,
            1.0,
        ),
        ("1300 dogs", "1,300 dogs.", 1.0, 1.0),
        ("25 percent", "25%", 1.0, 1.0),
        ("13.7 years", "13.7 years (SD 2.1).", 1.0, 1.0),
        # figures inside ordinary parentheses are still answers
        (
            "12.71 years for the Westie",
            "Longest: West Highland Terrier (12.71 years).",
            1.0,
            1.0,
        ),
        # a range is two positive numbers, not a negative one
        ("between 7.67 and 12.71", "7.67-12.71 years", 1.0, 1.0),
    ],
)
def test_numeric_metrics(pred, gold, primary, every):
    out = utils.process_results_numeric({"answer": gold}, [pred])
    assert out["numeric_acc"] == primary
    assert out["numeric_acc_all"] == every


def test_numbers_in_handles_separators_and_ranges():
    assert utils.numbers_in("1,300 dogs over 11.19-11.27 years") == [
        1300.0,
        11.19,
        11.27,
    ]
    assert utils.gold_numbers("11.23 years (95% CI 11.19-11.27).") == [11.23]


def test_fewshot_samples_have_the_dataset_fields():
    samples = utils.list_fewshot_samples()
    assert len(samples) == 3
    assert all({"question", "answer"} <= set(s) for s in samples)
