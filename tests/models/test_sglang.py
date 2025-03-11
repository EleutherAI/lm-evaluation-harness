from typing import List

import pytest
import torch

from lm_eval import evaluate, simple_evaluate, tasks
from lm_eval.api.instance import Instance
from lm_eval.tasks import get_task_dict


task_manager = tasks.TaskManager()


# We refer to vLLM's test but modify the trigger condition.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
# @pytest.mark.skip(reason="requires CUDA")
class Test_SGlang:
    sglang = pytest.importorskip("sglang")

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

            # NOTE(jinwei): EleutherAI/pythia-70m is not supported by SGlang yet. Instead we use Qwen models.
            cls.LM = SGLangLM(
                pretrained="Qwen/Qwen2-1.5B-Instruct",
                batch_size=1,
                tp_size=1,
                max_model_len=1024,
            )
        except Exception as e:
            pytest.fail(f"🔥 SGLangLM failed to initialize: {e}")

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

    # NOTE(Jinwei):A100 80GB is enough for our tests. If you run the last test "test_logliklihood_rolling" and OOM happens, please reduce the "max_model_len".
    def test_logliklihood_rolling(self) -> None:
        res = self.LM.loglikelihood_rolling(self.ROLLING)
        for x in res:
            assert isinstance(x, float)

    # def test_simple_evaluate(self)-> None:
    #     results = simple_evaluate(
    #         model =self.LM,
    #         tasks=["arc_easy"],
    #         # num_fewshot=0,
    #         task_manager=task_manager,
    #         limit= 10,
    #     )
    #     print(results)
    #     accuracy = results["results"]["arc_easy"]["acc,none"]
    #     print(f"Accuracy: {accuracy}")

    # def test_evaluate(self)-> None:
    #     tasks=["arc_easy"]
    #     task_dict = get_task_dict(tasks, task_manager)
    #     results = evaluate(
    #     lm=self.LM,
    #     task_dict=task_dict,
    #     limit= 10,
    #     )
    #     print(results)
    #     accuracy = results["results"]["arc_easy"]["acc,none"]
    #     print(f"Accuracy: {accuracy}")

    # TODO(jinwei): find out the outpt differences for "gsm_8k" with simple_evalute() and evaluate(). There are some errors in parser as well.
    def test_evaluator(self) -> None:
        simple_results = simple_evaluate(
            model=self.LM,
            tasks=["arc_easy"],
            task_manager=task_manager,
            limit=10,
        )
        assert simple_results is not None, "simple_evaluate returned None"
        # The accuracy for 10 data points is 0.7. Setting up a threshold of 0.5 provides a buffer to account for these fluctuations.
        assert simple_results["results"]["arc_easy"]["acc,none"] >= 0.5, (
            "The accuracy for simple_evaluate() is below 0.5!"
        )
        task_dict = get_task_dict(["arc_easy"], task_manager)
        evaluate_results = evaluate(
            lm=self.LM,
            task_dict=task_dict,
            limit=10,
        )
        assert evaluate_results is not None, "evaluate returned None"
        # The accuracy for 10 data points is 0.7. Setting up a threshold of 0.5 provides a buffer to account for these fluctuations.
        assert evaluate_results["results"]["arc_easy"]["acc,none"] >= 0.5, (
            "The accuracy for evaluate() is below 0.5!"
        )

        assert set(simple_results["results"].keys()) == set(
            evaluate_results["results"].keys()
        ), "Mismatch in task keys between simple_evaluate and evaluate"

        for task in simple_results["results"]:
            assert (
                simple_results["results"][task] == evaluate_results["results"][task]
            ), f"Mismatch in results for {task}"

        print(
            "✅ test_evaluator passed: simple_evaluate and evaluate results are identical."
        )
