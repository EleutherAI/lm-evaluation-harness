# ruff: noqa
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from lm_eval import tasks
from lm_eval.api.instance import Instance

pytest.skip("dependency conflict on CI", allow_module_level=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
task_manager = tasks.TaskManager()

TEST_STRING = "foo bar"


class Test_SteeredModel:
    from lm_eval.models.hf_steered import SteeredModel

    torch.use_deterministic_algorithms(True)
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
    version_minor = sys.version_info.minor
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)  # fewshot random generator seed
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: list[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances

    MULTIPLE_CH_RES = [
        -41.79737854003906,
        -42.964412689208984,
        -33.909732818603516,
        -37.055198669433594,
        -22.980390548706055,
        -20.268718719482422,
        -14.76205062866211,
        -27.887500762939453,
        -15.797225952148438,
        -15.914306640625,
        -13.01901626586914,
        -18.053699493408203,
        -13.33236312866211,
        -13.35921859741211,
        -12.12301254272461,
        -11.86703109741211,
        -47.02234649658203,
        -47.69982147216797,
        -36.420310974121094,
        -50.065345764160156,
        -16.742475509643555,
        -18.542402267456055,
        -26.460208892822266,
        -20.307228088378906,
        -17.686725616455078,
        -21.752883911132812,
        -33.17183303833008,
        -39.21712112426758,
        -14.78198528289795,
        -16.775150299072266,
        -11.49817180633545,
        -15.404842376708984,
        -13.141255378723145,
        -15.870940208435059,
        -15.29050064086914,
        -12.36030387878418,
        -44.557891845703125,
        -55.43851089477539,
        -52.66646194458008,
        -56.289222717285156,
    ]
    generate_until_RES = [
        " The average of $2.50 each is $",
        " A robe takes 2 bolts of blue fiber and half",
        " $50,000 in repairs.\n\nQuestion",
        " He runs 1 sprint 3 times a week.",
        " They feed each of her chickens three cups of mixed",
        " The price of the glasses is $5, but",
        " The total percentage of students who said they like to",
        " Carla is downloading a 200 GB file. Normally",
        " John drives for 3 hours at a speed of 60",
        " Eliza sells 4 tickets to 5 friends so she",
    ]
    ROLLING_RES = [
        -3604.61328125,
        -19778.67626953125,
        -8835.119384765625,
        -27963.37841796875,
        -7636.4351806640625,
        -9491.43603515625,
        -41047.35205078125,
        -8396.804443359375,
        -45966.24645996094,
        -7159.05322265625,
    ]
    LM = SteeredModel(
        pretrained="EleutherAI/pythia-70m",
        device="cpu",
        dtype="float32",
        steer_path="tests/testconfigs/sparsify_intervention.csv",
    )

    def test_load_with_sae_lens(self) -> None:
        from lm_eval.models.hf_steered import SteeredModel

        SteeredModel(
            pretrained="EleutherAI/pythia-70m",
            device="cpu",
            dtype="float32",
            steer_path="tests/testconfigs/sae_lens_intervention.csv",
        )

        assert True

    def test_loglikelihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        _RES, _res = self.MULTIPLE_CH_RES, [r[0] for r in res]
        # log samples to CI
        dir_path = Path("test_logs")
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"outputs_log_{self.version_minor}.txt"
        file_path = file_path.resolve()
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in _res))
        assert np.allclose(_res, _RES, atol=1e-2)
        # check indices for Multiple Choice
        argmax_RES, argmax_res = (
            np.argmax(np.array(_RES).reshape(-1, 4), axis=1),
            np.argmax(np.array(_res).reshape(-1, 4), axis=1),
        )
        assert (argmax_RES == argmax_res).all()

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until)
        assert res == self.generate_until_RES

    def test_loglikelihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        assert np.allclose(res, self.ROLLING_RES, atol=1e-1)

    def test_toc_encode(self) -> None:
        res = self.LM.tok_encode(TEST_STRING)
        assert res == [12110, 2534]

    def test_toc_decode(self) -> None:
        res = self.LM.tok_decode([12110, 2534])
        assert res == TEST_STRING

    def test_batch_encode(self) -> None:
        res = self.LM.tok_batch_encode([TEST_STRING, "bar foo"])[0].tolist()
        assert res == [[12110, 2534], [2009, 17374]]

    def test_model_generate(self) -> None:
        context = self.LM.tok_batch_encode([TEST_STRING])[0]
        res = self.LM._model_generate(context, max_length=10, stop=["\n\n"])
        res = self.LM.tok_decode(res[0])
        assert res == "foo bar\n<bazhang> !info bar"
