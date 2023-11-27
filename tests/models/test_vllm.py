import pytest
from typing import List
from lm_eval.api.instance import Instance
import lm_eval.tasks as tasks
import sys
import torch


@pytest.mark.skip(reason="requires CUDA")
class TEST_VLLM:
    vllm = pytest.importorskip("vllm")
    try:
        from lm_eval.models.vllm_causallms import VLLM

        LM = VLLM(pretrained="EleutherAI/pythia-70m")
    except ModuleNotFoundError:
        pass
    torch.use_deterministic_algorithms(True)
    tasks.initialize_tasks()
    multiple_choice_task = tasks.TASK_REGISTRY.get("arc_easy")()  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: List[Instance] = multiple_choice_task.instances
    generate_until_task = tasks.TASK_REGISTRY.get("gsm8k")()  # type: ignore
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until: List[Instance] = generate_until_task.instances
    rolling_task = tasks.TASK_REGISTRY.get("wikitext")()  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: List[Instance] = rolling_task.instances

    # TODO: make proper tests
    def test_logliklihood(self) -> None:
        res = self.LM.loglikelihood(self.MULTIPLE_CH)
        assert len(res) == len(self.MULTIPLE_CH)
        for x in res:
            assert isinstance(x[0], float)

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until)
        assert len(res) == len(self.generate_until)
        for x in res:
            assert isinstance(x, str)

    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        for x in res:
            assert isinstance(x, float)
