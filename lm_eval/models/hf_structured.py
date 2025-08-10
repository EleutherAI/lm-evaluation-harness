import logging

import xgrammar as xgr

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


eval_logger = logging.getLogger(__name__)

ALL_GRAMMAR_TYPES = ("gbnf", "json", "regex")


@register_model("hf-structured")
class HFStructuredLM(HFLM):
    """
    An abstracted Hugging Face model class for structured LMs.
    """

    def _get_logits_processor(self, grammar_file_path, grammar_type):
        if grammar_type not in ALL_GRAMMAR_TYPES:
            raise ValueError(
                f"Got invalid grammar_type '{grammar_type}', must be in '{','.join(ALL_GRAMMAR_TYPES)}'"
            )

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.config.vocab_size
        )
        compiler = xgr.GrammarCompiler(tokenizer_info)

        with open(grammar_file_path, "r") as f:
            grammar_str = f.read().strip()

        if grammar_type == "gbnf":
            compiled_grammar = compiler.compile_grammar(grammar_str)
        elif grammar_type == "json":
            compiled_grammar = compiler.compile_json_schema(grammar_str)
        elif grammar_type == "regex":
            compiled_grammar = compiler.compile_regex(grammar_str)

        return xgr.contrib.hf.LogitsProcessor(compiled_grammar)

    def _model_generate(
        self,
        context,
        max_length,
        stop,
        grammar_file_path,
        grammar_type,
        **generation_kwargs,
    ):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        logits_processor = self._get_logits_processor(grammar_file_path, grammar_type)

        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            logits_processor=[logits_processor],
            **generation_kwargs,
        )
