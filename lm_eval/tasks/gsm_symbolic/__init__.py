from lm_eval.api.task import TaskConfig, get_task_dict
from lm_eval.tasks.gsm_symbolic.gsm_symbolic import *
from lm_eval.tasks.gsm_symbolic.gsm_symbolic_cot import *

CONFIG = TaskConfig(
    name="gsm_symbolic",
    description="GSM-Symbolic math word problems dataset",
    keywords=["math", "word problems", "symbolic reasoning"],
    metrics=["exact_match"],
)

CONFIG_COT = TaskConfig(
    name="gsm_symbolic_cot",
    description="GSM-Symbolic math word problems dataset with Chain-of-Thought prompting",
    keywords=["math", "word problems", "symbolic reasoning", "chain of thought"],
    metrics=["exact_match"],
)