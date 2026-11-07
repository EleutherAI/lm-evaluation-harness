"""Evaluation files should use their format's encoding, not the host locale."""

import argparse
import json

import pytest
import yaml

from lm_eval.config.evaluate_config import EvaluatorConfig


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
@pytest.mark.parametrize("from_cli", [False, True])
def test_unicode_yaml_config(tmp_path, encoding, from_cli):
    payload = {"tasks": ["arc_easy"], "system_instruction": "请用中文回答。"}
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_bytes(
        yaml.safe_dump(payload, allow_unicode=True).encode(encoding)
    )

    if from_cli:
        config = EvaluatorConfig.from_cli(argparse.Namespace(config=str(config_path)))
    else:
        config = EvaluatorConfig.from_config(config_path)

    assert config.tasks == payload["tasks"]
    assert config.system_instruction == payload["system_instruction"]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16"])
def test_unicode_samples_file(tmp_path, encoding):
    samples = {"自定义任务": [2, 0]}
    samples_path = tmp_path / "samples.json"
    samples_path.write_bytes(json.dumps(samples, ensure_ascii=False).encode(encoding))

    config = EvaluatorConfig.from_cli(
        argparse.Namespace(tasks=["自定义任务"], samples=str(samples_path))
    )

    assert config.samples == samples


def test_invalid_yaml_file_still_reports_config_error(tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_bytes(b"tasks: [")

    with pytest.raises(ValueError, match="Invalid YAML"):
        EvaluatorConfig.from_config(config_path)


def test_invalid_samples_file_still_reports_json_error(tmp_path):
    samples_path = tmp_path / "invalid.json"
    samples_path.write_bytes(b'{"arc_easy": [}')

    with pytest.raises(json.JSONDecodeError):
        EvaluatorConfig.from_cli(
            argparse.Namespace(tasks=["arc_easy"], samples=str(samples_path))
        )
