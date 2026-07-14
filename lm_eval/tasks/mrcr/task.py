import logging
from copy import deepcopy

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask


eval_logger = logging.getLogger(__name__)


class MRCR(ConfigurableTask):
    def __init__(self, *args, config=None, **kwargs):
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        gen_prefix=None,
    ):
        if num_fewshot:
            eval_logger.warning(
                "MRCR ignores num_fewshot because prompts come directly from the source dataset."
            )
        if system_instruction:
            eval_logger.warning(
                "MRCR ignores system_instruction because prompts come directly from the source dataset."
            )
        if apply_chat_template or fewshot_as_multiturn or chat_template or gen_prefix:
            eval_logger.warning(
                "MRCR sends raw chat messages to the server; chat templating flags are ignored."
            )
        return doc["messages"]

    def construct_requests(
        self,
        doc,
        ctx,
        chat_template=None,
        apply_chat_template=False,
        **kwargs,
    ):
        gen_kwargs = deepcopy(self.config.generation_kwargs or {})
        requested_max_tokens = gen_kwargs.pop(
            "max_tokens",
            gen_kwargs.pop("max_gen_toks", None),
        )
        max_tokens = int(doc["max_generation_tokens"])
        if requested_max_tokens is not None:
            max_tokens = min(max_tokens, int(requested_max_tokens))

        return Instance(
            request_type="generate_until",
            doc=doc,
            arguments=(ctx, {**gen_kwargs, "max_tokens": max_tokens}),
            idx=0,
            **kwargs,
        )
