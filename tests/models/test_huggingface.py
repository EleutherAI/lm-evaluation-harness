from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import tokenizers
import torch
from packaging.version import parse as parse_version

from lm_eval import tasks
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"
task_manager = tasks.TaskManager()

TEST_STRING = "foo bar"


class Test_HFLM:
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
        -42.939308166503906,
        -33.914180755615234,
        -37.07139205932617,
        -22.95258331298828,
        -20.342208862304688,
        -14.818366050720215,
        -27.942853927612305,
        -15.80704116821289,
        -15.936427116394043,
        -13.052018165588379,
        -18.04828453063965,
        -13.345029830932617,
        -13.366025924682617,
        -12.127134323120117,
        -11.872495651245117,
        -47.10598373413086,
        -47.76410675048828,
        -36.4406852722168,
        -50.0289421081543,
        -16.72093963623047,
        -18.535587310791016,
        -26.46993637084961,
        -20.355995178222656,
        -17.757919311523438,
        -21.80595588684082,
        -33.1990852355957,
        -39.28636932373047,
        -14.759679794311523,
        -16.753942489624023,
        -11.486852645874023,
        -15.42177677154541,
        -13.15798282623291,
        -15.887393951416016,
        -15.28614616394043,
        -12.339089393615723,
        -44.59441375732422,
        -55.40888214111328,
        -52.70050811767578,
        -56.25089645385742,
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
        -3603.6328125,
        -19779.23974609375,
        -8834.16455078125,
        -27967.591796875,
        -7636.794982910156,
        -9491.93505859375,
        -41043.4248046875,
        -8397.689819335938,
        -45969.47155761719,
        -7158.90625,
    ]
    LM = HFLM(pretrained="EleutherAI/pythia-70m", device="cpu", dtype="float32")

    def test_logliklihood(self) -> None:
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

    def test_logliklihood_rolling(self) -> None:
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
        if parse_version(tokenizers.__version__) >= parse_version("0.20.0"):
            assert res == "foo bar\n<bazhang> !info bar"
        else:
            assert res == "foo bar\n<bazhang>!info bar"
