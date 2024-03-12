import pytest

import lm_eval.__main__


@pytest.fixture()
def parser():
    yield lm_eval.__main__.setup_parser()


def test_model(parser):
    result = parser.parse_args(["--model", "hf"])
    assert "hf" == result.model


def test_model_args(parser):
    result = parser.parse_args(["--model_args", "pretrained=EleutherAI/gpt-j-6B"])
    assert "pretrained=EleutherAI/gpt-j-6B" == result.model_args


def test_num_fewshot(parser):
    result = parser.parse_args(["--num_fewshot", "5"])
    assert 5 == result.num_fewshot


def test_tasks(parser):
    result = parser.parse_args(["--tasks", "hellaswag"])
    assert "hellaswag" == result.tasks


def test_trust_remote_code(parser):
    result = parser.parse_args(["--trust_remote_code"])
    assert result.trust_remote_code is True
