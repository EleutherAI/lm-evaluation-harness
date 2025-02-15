from typing import List

import pytest
import torch

from lm_eval import tasks
from lm_eval.api.instance import Instance


task_manager = tasks.TaskManager()


# If you just want to run the two tests, one A100 80GB is enough. When we ran the last test "test_logliklihood_rolling", OOM happened.
# Note(jinwei): we refer to vLLM's test but modify the trigger condition.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# @pytest.mark.skip(reason="requires CUDA")
class Test_SGlang:
    sglang = pytest.importorskip("sglang")
    # try:
    #     from lm_eval.models.sglang_causallms import SGLangLM

    #     LM = SGLangLM(pretrained="EleutherAI/pythia-70m",batch_size=128)
    #     # LM = SGLangLM(pretrained="Qwen/Qwen2-1.5B-Instruct")
    # except ModuleNotFoundError:
    #     pass

    # torch.use_deterministic_algorithms(True)
    task_list = task_manager.load_task_or_group(["arc_easy", "gsm8k", "wikitext"])
    multiple_choice_task = task_list["arc_easy"]  # type: ignore
    multiple_choice_task.build_all_requests(limit=10, rank=0, world_size=1)
    MULTIPLE_CH: List[Instance] = multiple_choice_task.instances
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until: List[Instance] = generate_until_task.instances
    rolling_task = task_list["wikitext"]  # type: ignore
    rolling_task.build_all_requests(limit=10, rank=0, world_size=1)
    ROLLING: List[Instance] = rolling_task.instances

    @classmethod
    def setup_class(cls):
        try:
            from lm_eval.models.sglang_causallms import SGLangLM

            # Note(jinwei): EleutherAI/pythia-70m is not supported by SGlang so I comment it here. Instead we use Qwen models.
            # cls.LM = SGLangLM(pretrained="EleutherAI/pythia-70m", batch_size=1)
            cls.LM = SGLangLM(
                pretrained="Qwen/Qwen2-1.5B-Instruct", batch_size=1, tp_size=1
            )
        except Exception as e:
            pytest.fail(f"🔥 SGLangLM failed to initialize: {e}")

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

    # Todo(jinwei): we set 4 A100 80GB gpus with tensor parallelism(tp_size=4) for this test, but when we ran the last test "test_logliklihood_rolling", OOM happened.
    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        for x in res:
            assert isinstance(x, float)
