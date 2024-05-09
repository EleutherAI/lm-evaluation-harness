import random

import transformers

from lm_eval import evaluator, tasks
from lm_eval.api.model import LM


class DryrunLM(LM):
    def __init__(self):
        super().__init__()
        self.tokencost = 0
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = "<|endoftext|>"

    @classmethod
    def create_from_arg_string(cls, arg_string):
        return cls()

    def loglikelihood(self, requests):
        res = []

        for ctx, cont in [req.args for req in requests]:
            res.append((-random.random(), False))
            self.tokencost += len(self.tokenizer.tokenize(ctx + cont))

        return res

    def generate_until(self, requests):
        res = []

        for ctx, _ in [req.args for req in requests]:
            res.append("lol")

            # assume worst case - generates until 256
            self.tokencost += len(self.tokenizer.tokenize(ctx)) + 256

        return res

    def loglikelihood_rolling(self, requests):
        res = []

        for (s,) in [req.args for req in requests]:
            res.append(-random.random())
            # assume worst case: extra full context
            self.tokencost += len(self.tokenizer.tokenize(s)) + 2048

        return res


def main():
    lm = DryrunLM()

    # task_list = "arc_challenge,arc_easy,boolq,cola,copa,headqa,hellaswag,lambada,logiqa,mathqa,mc_taco,mrpc,multirc,openbookqa,piqa,prost,pubmedqa,qnli,qqp,race,record,rte,sciq,sst,triviaqa,webqs,wic,wikitext,winogrande,wnli,wsc"
    task_list = "adv_glue,anli,beavertails,bigbench_bbq_lite_json_multiple_choice,discrim_eval,glue,lambada,logiqa,mc_taco,piqa,scruples,simple_cooccurrence_bias,toxigen,winogender,wmdp,xstest,bbh,hendrycks_ethics,moralchoice,decoding_trust,eq_bench,wikitext,pile_10k,bold"
    values = []
    for task_name in task_list.split(","):
        lm.tokencost = 0
        evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=0,
            limit=None,
            bootstrap_iters=10,
        )

        print(task_name, lm.tokencost)
        values.append(
            [
                task_name,
                lm.tokencost,
                lm.tokencost / 1000 * 0.0008,
                lm.tokencost / 1000 * 0.0012,
                lm.tokencost / 1000 * 0.006,
                lm.tokencost / 1000 * 0.06,
                lm.tokencost / 1000 * 0.0005,
                lm.tokencost / 1000 * 0.01
            ]
        )
    from pytablewriter import MarkdownTableWriter

    writer = MarkdownTableWriter()
    writer.headers = ["Task", "Tokens", "Ada", "Babbage", "Curie", "Davinci", "GPT3.5-Turbo", "GPT-4-Turbo"]

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
            totcost / 1000 * 0.0005,
            totcost / 1000 * 0.01,
        ]
    )

    writer.value_matrix = values

    print(writer.dumps())


if __name__ == "__main__":
    main()
