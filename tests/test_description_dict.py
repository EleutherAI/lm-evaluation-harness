import numpy as np
from promptsource.templates import DatasetTemplates

import lm_eval.tasks
import lm_eval.models


def test_description_dict():
    rng = np.random.default_rng(42)
    num_fewshot = 1

    task_to_prompt = {
        "axg": DatasetTemplates("super_glue", "axg")["can we infer"],
        "wnli": DatasetTemplates("glue", "wnli")["confident"],
    }
    description_dict = {
        "axg": "This task is used to measure  measure gender bias in coreference "
        "resolution systems. Follow the prompt instructions to complete the task",
        "wnli": "This task tests reading comprehension. Follow the prompt "
        "instructions to complete the task.",
    }

    task_dict = {
        task: lm_eval.tasks.get_task(task)(prompt_template=prompt)
        for task, prompt in task_to_prompt.items()
    }
    for task_name, task in task_dict.items():
        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )
        docs = task.evaluation_docs()
        for _, doc in (
            zip(range(num_fewshot), docs) if num_fewshot > 0 else enumerate(docs)
        ):
            ctx = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                description=description,
                rng=rng,
            )[0]
            assert description in ctx
