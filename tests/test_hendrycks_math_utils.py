from lm_eval.tasks.hendrycks_math.utils import _extract_model_answer


def test_extract_model_answer_from_dollar_delimiters() -> None:
    prediction = "The answer is $42$"
    assert _extract_model_answer(prediction) == "42"


def test_extract_model_answer_from_boxed_without_dollars() -> None:
    prediction = "Thus the answer is \\[ \\boxed{42} \\]"
    assert _extract_model_answer(prediction) == "42"


def test_extract_model_answer_returns_prediction_when_unparseable() -> None:
    prediction = "no structured answer here"
    assert _extract_model_answer(prediction) == prediction
