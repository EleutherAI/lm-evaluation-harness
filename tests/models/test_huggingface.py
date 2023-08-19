from __future__ import annotations
import pytest
import numpy as np
from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance
import lm_eval.tasks as tasks


class Test_HFLM:

    multiple_choice_task = tasks.TASK_REGISTRY.get("arc_easy")()  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: list[Instance] = multiple_choice_task.instances
    greedy_until_task = tasks.TASK_REGISTRY.get("gsm8k_yaml")()  # type: ignore
    greedy_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    greedy_until_task._config.generation_kwargs["max_gen_toks"] = 10
    GREEDY_UNTIL: list[Instance] = greedy_until_task.instances
    rolling_task = tasks.TASK_REGISTRY.get("wikitext")()  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: list[Instance] = rolling_task.instances

    MULTIPLE_CH_RES = [
        (-41.905879974365234, False),
        (-42.93785095214844, False),
        (-33.9145393371582, False),
        (-37.07110595703125, False),
        (-22.954187393188477, False),
        (-20.342954635620117, False),
        (-14.816370010375977, False),
        (-27.94381332397461, False),
        (-15.806619644165039, False),
        (-15.937178611755371, False),
        (-13.052162170410156, False),
        (-18.04889678955078, False),
        (-13.346054077148438, False),
        (-13.367782592773438, False),
        (-12.128646850585938, False),
        (-11.871688842773438, False),
        (-47.10654067993164, False),
        (-47.76068115234375, False),
        (-36.44114303588867, False),
        (-50.02851104736328, False),
        (-16.719867706298828, False),
        (-18.537654876708984, False),
        (-26.469972610473633, False),
        (-20.356552124023438, False),
        (-17.75723648071289, False),
        (-21.8068790435791, False),
        (-33.19971466064453, False),
        (-39.2862434387207, False),
        (-14.762389183044434, False),
        (-16.75531005859375, False),
        (-11.486998558044434, False),
        (-15.421247482299805, False),
        (-13.157613754272461, False),
        (-15.88864517211914, False),
        (-15.287158012390137, False),
        (-12.339122772216797, False),
        (-44.59400177001953, False),
        (-55.40974807739258, False),
        (-52.697017669677734, False),
        (-56.252601623535156, False),
    ]
    GREEDY_UNTIL_RES = [
        " The average of $2.50 each is $",
        " A robe takes 2 bolts of blue fiber and half",
        " $50,000 in repairs.",
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
        _RES, _res = [r[0] for r in self.MULTIPLE_CH_RES], [r[0] for r in res]
        print(_res)
        assert np.allclose(_res, _RES, atol=1e-2)
        # check indices for Multiple Choice
        argmax_RES, argmax_res = np.argmax(
            np.array(_RES).reshape(-1, 4), axis=1
        ), np.argmax(np.array(_res).reshape(-1, 4), axis=1)
        assert (argmax_RES == argmax_res).all()

    def test_greedy_until(self) -> None:
        res = self.LM.greedy_until(self.GREEDY_UNTIL)
        assert res == self.GREEDY_UNTIL_RES

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        assert np.allclose(res, self.ROLLING_RES, atol=1e-2)

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
