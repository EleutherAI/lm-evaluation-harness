import copy

import torch
import transformers
from tqdm import tqdm
from transformers import BatchEncoding

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import (
    Collator,
    replace_placeholders,
)
from lm_eval.models.utils_hf import stop_sequences_criteria


DEFAULT_AUDIO_PLACEHOLDERS = ["<audio>"]


@register_model("hf-audiolm-qwen")
class HFAUDIOLMQWEN(HFLM):
    """
    An abstracted Hugging Face model class for Audio LM model like Qwen2-Audio.
    """

    AUTO_MODEL_CLASS = transformers.Qwen2AudioForConditionalGeneration
    MULTIMODAL = True  # flag to indicate, for now, that this model type can run multimodal requests

    def __init__(
        self,
        pretrained: str | transformers.PreTrainedModel,
        max_audios: int | None = 5,
        **kwargs,
    ):
        # We initialize using HFLM's init. Sub-methods like _create_model and _create_tokenizer
        # modify init behavior.
        super().__init__(pretrained, **kwargs)
        self.max_audios = max_audios
        self.chat_applied: bool = False

    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str | transformers.ProcessorMixin | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        **kwargs,
    ) -> None:
        """
        Helper method during initialization.
        For the multimodal variant, we initialize not just
        `self.tokenizer` but also `self.processor`.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                return transformers.AutoTokenizer.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    # use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.ProcessorMixin
                )  # TODO: check this condition
                return tokenizer

        # Get tokenizer based on 'pretrained'
        if isinstance(pretrained, str):
            model_name = pretrained
        else:
            # get the HF hub name via accessor on model
            model_name = self.model.name_or_path

        self.processor = transformers.AutoProcessor.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # use_fast=use_fast_tokenizer,
        )
        self.tokenizer = self.processor.tokenizer

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """

        chat_templated = self.processor.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt
        )

        return chat_templated

    def _model_multimodal_generate(self, inputs, max_length, stop, **generation_kwargs):
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        return self.model.generate(
            **inputs,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )

    def tok_batch_multimodal_encode(
        self,
        strings: list[str],  # note that input signature of this fn is different
        audios: list[list],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> (
        BatchEncoding | dict[str, torch.Tensor]
    ):  # note that this return signature differs from HFLM tok_batch_encode.
        # NOTE: here, we replace <audio> tags with our model's corresponding image_token string value.
        def _replace_placeholder(placeholder, strings):
            return [
                replace_placeholders(
                    string,
                    placeholder,
                    "<|audio_bos|><|AUDIO|><|audio_eos|>",
                    self.max_audios,
                )
                for string in strings
            ]

        if not self.chat_applied:
            # TODO<baber>: This still keeps the whitespace in the image placeholder, which is not ideal.
            for placeholder in DEFAULT_AUDIO_PLACEHOLDERS:
                strings = _replace_placeholder(placeholder, strings)

        encoding = self.processor(
            audios=audios,
            text=strings,
            padding=True,
            return_tensors="pt",
            # **add_special_tokens, # TODO: at least some Processors error out when passing this. How do we control whether text gets BOS added?
        )

        encoding.to(  # TODO: our other tokenization methods in HFLM don't typically move to device. this breaks convention
            self.device, self.model.dtype
        )  # TODO: This only casts the pixel values. Should they always be float16?

        return encoding

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests with text+audio input",
        )
        # TODO: port auto-batch sizing into this.

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = Collator(
            [reg.args for reg in requests],
            _collate,
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        ### Up to here: was identical to non-multimodal HFLM generate_until ###

        for chunk in chunks:
            contexts, all_gen_kwargs, aux_arguments = zip(*chunk, strict=False)

            audios = []
            for audio_lst_dict in aux_arguments:
                for audio in audio_lst_dict["audio"]:
                    audios.append(audio["array"])

            if not isinstance(contexts, list):
                contexts = list(
                    contexts
                )  # for Qwen2-VL, processor is unhappy accepting a tuple of strings instead of a list.
                # TODO: could we upstream this workaround to HF?
            ### this part onward: same as HFLM ###

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            until = None
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                if "until" in kwargs.keys():
                    until = kwargs.pop("until")
                    if isinstance(until, str):
                        until = [until]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                )
            # add EOS token to stop sequences
            eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)
            if not until:
                until = [eos]
            else:
                until.append(eos)
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            ## end stuff that's entirely copied verbatim from HFLM ###

            max_ctx_len = self.max_length - max_gen_toks

            inputs = self.tok_batch_multimodal_encode(
                contexts,
                audios,
                left_truncate_len=max_ctx_len,
                truncation=self.truncation,
            )

            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs.input_ids = inputs.input_ids.to("cuda")
            cont = self._model_multimodal_generate(inputs, stop=until, **kwargs)

            del inputs
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            ### essentially same as HFLM beyond this line!

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts, strict=False):
                # discard context + left-padding toks if using causal decoder-only VLM
                cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                res.append(s)
                self.cache_hook.add_partial(
                    "generate_until", (context, gen_kwargs), s
                )  # TODO: cache key for multimodal input should be what?
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError(
            "model type `hf-audiolm` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks ",
            "this is because we do not support measuring the loglikelihood a model assigns to an image.",
        )

    def loglikelihood(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "'loglikelihood' requests for model type `hf-audiolm` are not yet tested. This feature will be enabled when a loglikelihood-based multiple-choice VQA dataset is added!"
        )
