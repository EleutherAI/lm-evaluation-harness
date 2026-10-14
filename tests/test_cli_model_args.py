import pytest

from lm_eval import evaluator


@pytest.mark.parametrize(
    "flags,yaml_config,expected",
    [
        (
            [
                "--model_args",
                "pretrained=x,batch_size=32,device=cuda:1,max_batch_size=64",
            ],
            None,
            {
                "pretrained": "x",
                "batch_size": 32,
                "device": "cuda:1",
                "max_batch_size": 64,
            },
        ),
        (
            ["--model_args", "pretrained=x,batch_size=32", "--batch_size", "8"],
            None,
            {"pretrained": "x", "batch_size": "8", "device": "cuda:0"},
        ),
        (
            ["--model_args", "pretrained=x,batch_size=32", "--batch_size", "1"],
            None,
            {"pretrained": "x", "batch_size": "1", "device": "cuda:0"},
        ),
        (
            ["--model_args", "pretrained=x", "--batch_size", "8"],
            None,
            {"pretrained": "x", "batch_size": "8", "device": "cuda:0"},
        ),
        (
            [
                "--model_args",
                "device=cpu,max_batch_size=64",
                "--device",
                "cuda:0",
                "--max_batch_size",
                "8",
            ],
            None,
            {"batch_size": 1, "device": "cuda:0", "max_batch_size": 8},
        ),
        (
            ["--model_args", "batch_size=32", "--batch_size", "auto:2"],
            None,
            {"batch_size": "auto:2", "device": "cuda:0"},
        ),
        (
            [],
            "model_args: {batch_size: 32, device: 'cuda:1'}\n",
            {"batch_size": 32, "device": "cuda:1"},
        ),
        (
            [],
            "model_args: {batch_size: 32, device: 'cuda:1'}\nbatch_size: 4\ndevice: cpu\n",
            {"batch_size": 4, "device": "cpu"},
        ),
        (
            ["--batch_size", "8"],
            "model_args: {batch_size: 32}\nbatch_size: 4\n",
            {"batch_size": "8", "device": "cuda:0"},
        ),
        ([], None, {"batch_size": 1, "device": "cuda:0"}),
    ],
)
def test_cli_model_args_preserve_explicit_configuration(
    monkeypatch, tmp_path, flags, yaml_config, expected
):
    import argparse

    from lm_eval._cli.run import Run
    from lm_eval.api.model import LM
    from lm_eval.config.evaluate_config import EvaluatorConfig
    from lm_eval.models.dummy import DummyLM

    class RecordingDummyLM(DummyLM):
        def __init__(self, **kwargs):
            self.received_kwargs = kwargs.copy()
            super().__init__(**kwargs)

        @classmethod
        def create_from_arg_string(cls, arg_string, additional_config=None):
            return LM.create_from_arg_string.__func__(
                cls, arg_string, additional_config
            )

    class EmptyTaskManager:
        def __init__(self):
            self.task_index = {}

        def load(self, tasks):
            assert tasks == ["test_task"]
            return {"tasks": {}, "groups": {}}

    captured = {}

    def fake_evaluate(lm, **kwargs):
        captured["lm"] = lm
        return {"results": {}}

    monkeypatch.setattr(
        evaluator.lm_eval.api.registry, "get_model", lambda name: RecordingDummyLM
    )
    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluator, "add_env_info", lambda results: None)
    monkeypatch.setattr(evaluator, "add_tokenizer_info", lambda results, lm: None)
    parser = argparse.ArgumentParser()
    Run.create(parser.add_subparsers())
    if yaml_config is not None:
        config_path = tmp_path / "eval.yaml"
        config_path.write_text(yaml_config)
        flags = ["--config", str(config_path), *flags]
    cfg = EvaluatorConfig.from_cli(
        parser.parse_args(["run", "--model", "dummy", *flags])
    )
    original_model_args = cfg.model_args.copy()
    result = evaluator.simple_evaluate(
        model=cfg.model,
        model_args=cfg.model_args,
        tasks=["test_task"],
        batch_size=cfg.batch_size,
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        task_manager=EmptyTaskManager(),
        random_seed=None,
        numpy_random_seed=None,
        torch_random_seed=None,
        fewshot_random_seed=None,
    )
    assert result is not None
    assert captured["lm"].received_kwargs == expected
    assert cfg.model_args == original_model_args
