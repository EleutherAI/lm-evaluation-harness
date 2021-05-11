import json
import numpy as np
import random
import logging
from lm_eval import models, tasks, evaluator, base

logging.getLogger("openai").setLevel(logging.WARNING)


fewshot_descriptions = [
    "foo",
    "bar"
]

task = "lambada"
num_fewshot = 0
model = "gpt2"
model_args = ""
limit = None
no_cache = False


class CustomDescTask:
    def __init__(self, task, desc):
        self.task = task
        self.desc = desc

        def fewshot_description():
            return self.desc
        
        self.task.fewshot_description = fewshot_description

    def __getattr__(self, attr):
        return getattr(self.task, attr)


def main():
    random.seed(42)
    np.random.seed(42)

    lm = models.get_model(model).create_from_arg_string(model_args)
    
    if limit:
        print("WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if not no_cache:
        lm = base.CachingLM(lm, 'lm_cache/' + model + '_' + model_args.replace('=', '-').replace(',', '_') + '.db')

    task_dict = tasks.get_task_dict([task])

    for desc in fewshot_descriptions:
        custom_task_dict = {k: CustomDescTask(v, desc) for k, v in task_dict.items()}

        results = evaluator.evaluate(lm, custom_task_dict, True, num_fewshot, limit)

        dumped = json.dumps(results, indent=2)

        print('Description:', desc)
        print(dumped)

        # MAKE TABLE
        from pytablewriter import MarkdownTableWriter

        writer = MarkdownTableWriter()
        writer.headers = ["Task", "Metric", "Value"]

        values = []

        for k, dic in results.items():
            for m, v in dic.items():
                values.append([k, m, '%.4f' % v])
                k = ""
        writer.value_matrix = values

        print(writer.dumps())


if __name__ == "__main__":
    main()
