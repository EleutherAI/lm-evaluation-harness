import random
import transformers
from lm_eval import tasks, evaluator
from lm_eval.base import LM


class DryrunLM(LM):
    def __init__(self):
        self.tokencost = 0
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
        self.tokenizer.pad_token = "<|endoftext|>"

    @classmethod
    def create_from_arg_string(cls, arg_string):
        return cls()

    def loglikelihood(self, requests):
        res = []
        
        for ctx, cont in requests:
            res.append((-random.random(), False))
            self.tokencost += len(self.tokenizer.tokenize(ctx + cont))

        return res
    
    def greedy_until(self, requests):
        res = []
        
        for ctx, until in requests:
            res.append("lol")

            # assume worst case - generates until 256
            self.tokencost += len(self.tokenizer.tokenize(ctx)) + 256

        return res


def main():
    lm = DryrunLM()

    values = []
    for taskname in list(tasks.TASK_REGISTRY.keys()):
        lm.tokencost = 0
        evaluator.evaluate(lm, {taskname: tasks.get_task(taskname)()}, False, 0, None)

        print(taskname, lm.tokencost)
        values.append([taskname, lm.tokencost, lm.tokencost / 1000 * 0.06])
    from pytablewriter import MarkdownTableWriter

    writer = MarkdownTableWriter()
    writer.headers = ["Task", "Tokens", "Davinci Cost"]

    values.sort(key=lambda x: -x[1])
    totcost = sum([x[1] for x in values])
    values.append(["**Total**", totcost, totcost / 1000 * 0.06])

    writer.value_matrix = values

    print(writer.dumps())
if __name__ == "__main__":
    main()
