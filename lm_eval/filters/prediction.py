from typing import List

import lm_eval
from lm_eval.api.filter import Filter
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_filter
from lm_eval.utils import eval_logger, simple_parse_args_string


@register_filter("lmjudge")
class LmJudge(Filter):
    def __init__(
        self, model, model_args, batch_size, max_batch_size, device, reqtype
    ) -> None:
        eval_logger.info(
            f"Initializing {model} model, with arguments: {simple_parse_args_string(model_args)}"
        )
        self.lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
        self.reqtype = reqtype

    def apply(self, resps, docs) -> List[List[str]]:
        resps = [
            Instance(
                request_type="generate_until",
                doc=None,
                arguments=(x[0], {"max_gen_toks": 10}),
                idx=i,
            )
            for i, x in enumerate(resps)
        ]

        resps = getattr(self.lm, self.reqtype)(resps)
        return [[x] for x in resps]
