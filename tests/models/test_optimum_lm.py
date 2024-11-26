from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

import lm_eval.tasks as tasks
from lm_eval.api.instance import Instance
from lm_eval.models.optimum_lm import OptimumLM


task_manager = tasks.TaskManager()


class Test_OptimumLM:
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
        -41.902435302734375,
        -42.93859100341797,
        -33.91422653198242,
        -37.07378387451172,
        -22.952665328979492,
        -20.342262268066406,
        -14.821088790893555,
        -27.94265365600586,
        -15.808233261108398,
        -15.9349365234375,
        -13.051279067993164,
        -18.048309326171875,
        -13.344562530517578,
        -13.365070343017578,
        -12.127277374267578,
        -11.871906280517578,
        -47.10643005371094,
        -47.76600646972656,
        -36.442317962646484,
        -50.03009033203125,
        -16.723222732543945,
        -18.538711547851562,
        -26.471467971801758,
        -20.3573055267334,
        -17.759742736816406,
        -21.806621551513672,
        -33.19993591308594,
        -39.28681182861328,
        -14.760702133178711,
        -16.75447654724121,
        -11.486654281616211,
        -15.42199420928955,
        -13.156449317932129,
        -15.888359069824219,
        -15.285746574401855,
        -12.339546203613281,
        -44.59436798095703,
        -55.41197204589844,
        -52.696617126464844,
        -56.25568771362305,
    ]

    generate_until_RES = [
        " The average of $2.50 each is about",
        " A robe takes 2 bolts of blue fiber and half",
        " $50,000 in repairs.\n\nQuestion",
        " He runs 3 sprints 3 times a week.",
        " They feed each of her chickens three cups of mixed",
        " The price of the glasses is $5.\n",
        " The total percentage of students who said they like to",
        " Carla is downloading a 200 GB file. Normally",
        " John drives for 3 hours at a speed of 60",
        " Eliza sells 4 tickets for a total of 20",
    ]

    ROLLING_RES = [
        -3603.674072265625,
        -19779.1611328125,
        -8834.160400390625,
        -27967.6396484375,
        -7636.7506103515625,
        -9491.9541015625,
        -41043.45849609375,
        -8397.63720703125,
        -45969.47253417969,
        -7158.84765625,
    ]

    LM = OptimumLM(pretrained="EleutherAI/pythia-70m", device="cpu", dtype="float32")

    def test_logliklihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        _RES, _res = self.MULTIPLE_CH_RES, [r[0] for r in res]
        # log samples to CI
        dir_path = Path("test_logs")
        dir_path.mkdir(parents=True, exist_ok=True)

        file_path = dir_path / f"outputs_log_{self.version_minor}.txt"
        file_path = file_path.resolve()
        with open(file_path, "w") as f:
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

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        assert np.allclose(res, self.ROLLING_RES, atol=1e-1)

    def test_toc_encode(self) -> None:
        res = self.LM.tok_encode("foo bar")
        assert res == [12110, 2534]

    def test_toc_decode(self) -> None:
        res = self.LM.tok_decode([12110, 2534])
        assert res == "foo bar"

    def test_batch_encode(self) -> None:
        res = self.LM.tok_batch_encode(["foo bar", "bar foo"])[0].tolist()
        assert res == [[12110, 2534], [2009, 17374]]

    def test_model_generate(self) -> None:
        context = self.LM.tok_batch_encode(["foo bar"])[0]
        res = self.LM._model_generate(context, max_length=10, stop=["\n\n"])
        res = self.LM.tok_decode(res[0])
        assert res == "foo bar\n<bazhang>!info bar"
