import sys

import lm_eval.models.anthropic_llms as anthropic_llms
import lm_eval.tasks as tasks
from lm_eval.api.instance import Instance
from lm_eval.models.anthropic_llms import (
    AnthropicChatLM,
    AnthropicLM,
)


def mock_anthropic_completion(**kwargs):
    prompt = kwargs["prompt"]
    model = kwargs["model"].strip()
    if model == "claude-2.0":
        if prompt.endswith(
            "How much in dollars does she make every day at the farmers' market?\nAnswer:"
        ):
            return " * Janet's ducks lay 16 eggs per"
        if prompt.endswith("How many bolts in total does it take?\nAnswer:"):
            return " * A robe takes 2 bolts of blue fiber"
        if prompt.endswith("How much profit did he make?\nAnswer:"):
            return " * Josh bought the house for $80,000"
        if prompt.endswith("How many total meters does he run a week?\nAnswer:"):
            return " * James runs 3 sprints per week\n*"
        if prompt.endswith("if the size of Wendi's flock is 20 chickens?\nAnswer:"):
            return " * Wendi has 20 chickens\n* Each"
        if prompt.endswith("How much does he need to pay for them?\nAnswer:"):
            return " * One glass costs $5\n* Every secon"
        if prompt.endswith(
            "do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?\nAnswer:"
        ):
            return " * Seattle has 20 sheep\n* Charleston has 4"
        if prompt.endswith("How load does it take to download the file?\nAnswer:"):
            return " * Carla can download 2 GB/minute"
        if prompt.endswith(
            "How far is he from home at the end of those 4 hours?\nAnswer:"
        ):
            return " * John originally drove for 3 hours at 60 mph"
        if prompt.endswith("how much are her earnings for this week?\nAnswer:"):
            return " * Eliza's regular rate is $10 per"


def mock_anthropic_chat(**kwargs):
    prompt = kwargs["prompt"]
    model = kwargs["model"].strip()
    if model == "claude-3-haiku-20240307":
        if prompt.endswith(
            "How much in dollars does she make every day at the farmers' market?\nAnswer:"
        ):
            return "Okay, let's break this down step-by-step:\n* Janet's ducks"
        if prompt.endswith("How many bolts in total does it take?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n* A robe takes"
        if prompt.endswith("How much profit did he make?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n1. Josh bought the"
        if prompt.endswith("How many total meters does he run a week?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n* James runs 3"
        if prompt.endswith("if the size of Wendi's flock is 20 chickens?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n* Wendi feeds"
        if prompt.endswith("How much does he need to pay for them?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n* Each glass costs $"
        if prompt.endswith(
            "do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?\nAnswer:"
        ):
            return "Okay, let's break this down step-by-step:\n\n1. Seattle has "
        if prompt.endswith("How load does it take to download the file?\nAnswer:"):
            return "To solve this problem, we need to calculate the time it takes to download the file before the restart"
        if prompt.endswith(
            "How far is he from home at the end of those 4 hours?\nAnswer:"
        ):
            return "Okay, let's break this down step-by-step:\n\n1. John drives for"
        if prompt.endswith("how much are her earnings for this week?\nAnswer:"):
            return "Okay, let's break this down step-by-step:\n\n1. Eliza"


# monkey patch the completion functions
anthropic_llms.anthropic_completion = mock_anthropic_completion
anthropic_llms.anthropic_chat = mock_anthropic_chat


class Test_AnthropicLM:
    version_minor = sys.version_info.minor
    task_manager = tasks.TaskManager()
    task_list = task_manager.load_task_or_group(["gsm8k"])
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 10
    generate_until_task.set_fewshot_seed(1234)
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until_instances: list[Instance] = generate_until_task.instances
    generate_until_results = [
        " * Janet's ducks lay 16 eggs per",
        " * A robe takes 2 bolts of blue fiber",
        " * Josh bought the house for $80,000",
        " * James runs 3 sprints per week\n*",
        " * Wendi has 20 chickens\n* Each",
        " * One glass costs $5\n* Every secon",
        " * Seattle has 20 sheep\n* Charleston has 4",
        " * Carla can download 2 GB/minute",
        " * John originally drove for 3 hours at 60 mph",
        " * Eliza's regular rate is $10 per",
    ]
    LM = AnthropicLM("claude-2.0")

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until_instances)
        assert res == self.generate_until_results

    def test_toc_encode(self) -> None:
        res = self.LM.tok_encode("foo bar")
        assert res == [3803, 3871]

    def test_toc_decode(self) -> None:
        res = self.LM.tok_decode([3803, 3871])
        assert res == "foo bar"


class Test_AnthropicChatLM:
    version_minor = sys.version_info.minor
    task_manager = tasks.TaskManager()
    task_list = task_manager.load_task_or_group(["gsm8k"])
    generate_until_task = task_list["gsm8k"]  # type: ignore
    generate_until_task._config.generation_kwargs["max_gen_toks"] = 20
    generate_until_task.set_fewshot_seed(1234)
    generate_until_task.build_all_requests(limit=10, rank=0, world_size=1)
    generate_until_instances: list[Instance] = generate_until_task.instances
    generate_until_results = [
        "Okay, let's break this down step-by-step:\n* Janet's ducks",
        "Okay, let's break this down step-by-step:\n* A robe takes",
        "Okay, let's break this down step-by-step:\n1. Josh bought the",
        "Okay, let's break this down step-by-step:\n* James runs 3",
        "Okay, let's break this down step-by-step:\n* Wendi feeds",
        "Okay, let's break this down step-by-step:\n* Each glass costs $",
        "Okay, let's break this down step-by-step:\n\n1. Seattle has ",
        "To solve this problem, we need to calculate the time it takes to download the file before the restart",
        "Okay, let's break this down step-by-step:\n\n1. John drives for",
        "Okay, let's break this down step-by-step:\n\n1. Eliza",
    ]

    LM = AnthropicChatLM("claude-3-haiku-20240307")

    def test_generate_until(self) -> None:
        res = self.LM.generate_until(self.generate_until_instances)
        assert res == self.generate_until_results
