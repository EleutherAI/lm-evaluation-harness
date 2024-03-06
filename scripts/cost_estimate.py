import random

import transformers

from lm_eval import evaluator, tasks
from lm_eval.api.model import LM


class DryrunLM(LM):
    def __init__(self):
        self.tokencost = 0
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
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

    def generate_until(self, requests):
        res = []

        for ctx, _ in requests:
            res.append("lol")

            # assume worst case - generates until 256
            self.tokencost += len(self.tokenizer.tokenize(ctx)) + 256

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for (s,) in requests:
            # assume worst case: extra full context
            self.tokencost += len(self.tokenizer.tokenize(s)) + 2048

        return res


def main():
    lm = DryrunLM()

    task_list = "arc_challenge,arc_easy,boolq,cola,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,record,rte,sciq,sst,triviaqa,webqs,wic,wikitext,winogrande,wnli,wsc"
    values = []
    for taskname in task_list.split(","):
        lm.tokencost = 0
        evaluator.simple_evaluate(
            lm=lm,
            task_dict={taskname: tasks.get_task(taskname)()},
            num_fewshot=0,
            limit=None,
            bootstrap_iters=10,
        )

        print(taskname, lm.tokencost)
        values.append(
            [
                taskname,
                lm.tokencost,
                lm.tokencost / 1000 * 0.0008,
                lm.tokencost / 1000 * 0.0012,
                lm.tokencost / 1000 * 0.006,
                lm.tokencost / 1000 * 0.06,
            ]
        )
    from pytablewriter import MarkdownTableWriter

    writer = MarkdownTableWriter()
    writer.headers = ["Task", "Tokens", "Ada", "Babbage", "Curie", "Davinci"]

    values.sort(key=lambda x: -x[1])
    totcost = sum([x[1] for x in values])
    values.append(
        [
            "**Total**",
            totcost,
            totcost / 1000 * 0.0008,
            totcost / 1000 * 0.0012,
            totcost / 1000 * 0.006,
            totcost / 1000 * 0.06,
        ]
    )

    writer.value_matrix = values

    print(writer.dumps())


if __name__ == "__main__":
    main()
