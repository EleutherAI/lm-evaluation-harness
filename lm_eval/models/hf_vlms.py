import copy
from typing import List, Optional, Tuple, Union

import transformers
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import Collator, stop_sequences_criteria


DEFAULT_IMAGE_TOKEN = "<image>"


@register_model("hf-multimodal")
class HFMultimodalLM(HFLM):
    """
    An abstracted Hugging Face model class for multimodal LMs like Llava and Idefics.
    """

    AUTO_MODEL_CLASS = transformers.AutoModelForVision2Seq

    # @property
    # def max_length(self):
    #     raise NotImplementedError

    @property
    def tokenizer_name(self) -> str:
        return self.processor.tokenizer.name_or_path.replace("/", "__")

    @property
    def chat_template(self) -> str:
        if self.processor.tokenizer.chat_template is not None:
            return self.processor.tokenizer.chat_template
        return self.processor.tokenizer.default_chat_template

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
    ) -> None:
        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

    # def _create_model(
    #     self,
    #     pretrained: Union[str, transformers.PreTrainedModel],
    #     revision="main",
    #     dtype="auto",
    #     trust_remote_code=False,
    #     **kwargs,
    # ) -> None:
    #     """
    #     Initializes an HF or HF-compatible PreTrainedModel from scratch
    #     inside HFLM, using the kwargs passed into self.__init__().
    #     """

    #     model_kwargs = kwargs if kwargs else {}

    #     if parallelize:
    #        # do stuff
    #        pass

    #     if isinstance(pretrained, str):

    #         return self.AUTO_MODEL_CLASS.from_pretrained(
    #             pretrained,
    #             revision=revision,
    #             torch_dtype=get_dtype(dtype),
    #             trust_remote_code=trust_remote_code,
    #             **model_kwargs,
    #         )

    #     assert isinstance(pretrained, transformers.PreTrainedModel)
    #     return pretrained

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.ProcessorMixin,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """
        Helper method during initialization.
        """

        if tokenizer:
            if isinstance(tokenizer, str):
                return transformers.AutoProcessor.from_pretrained(
                    tokenizer,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    # use_fast=use_fast_tokenizer,
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
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

    # def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
    #     """
    #     Method to apply a chat template to a list of chat history between user and model.
    #     """
    #     return self.tokenizer.apply_chat_template(
    #         chat_history, tokenize=False, add_generation_prompt=True
    #     )

    # def tok_encode(
    #     self, string: str, left_truncate_len=None, add_special_tokens=None
    # ) -> List[int]:
    #     """ """
    #     # default for None - empty dict, use predefined tokenizer param
    #     # used for all models except for CausalLM or predefined value
    #     special_tokens_kwargs = {}

    #     # by default for CausalLM - false or self.add_bos_token is set
    #     if add_special_tokens is None:
    #         if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
    #             special_tokens_kwargs = {
    #                 "add_special_tokens": False or self.add_bos_token
    #             }
    #     # otherwise the method explicitly defines the value
    #     else:
    #         special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

    #     encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

    #     # left-truncate the encoded context to be at most `left_truncate_len` tokens long
    #     if left_truncate_len:
    #         encoding = encoding[-left_truncate_len:]

    #     return encoding

    # def tok_batch_encode(
    #     self,
    #     strings: List[str],
    #     padding_side: str = "left",
    #     left_truncate_len: int = None,
    #     truncation: bool = False,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
    #     old_padding_side = self.tokenizer.padding_side
    #     self.tokenizer.padding_side = padding_side

    #     add_special_tokens = {}
    #     if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM:
    #         add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

    #     encoding = self.tokenizer(
    #         strings,
    #         truncation=truncation,
    #         padding="longest",
    #         return_tensors="pt",
    #         **add_special_tokens,
    #     )
    #     if left_truncate_len:
    #         encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
    #         encoding["attention_mask"] = encoding["attention_mask"][
    #             :, -left_truncate_len:
    #         ]
    #     self.tokenizer.padding_side = old_padding_side

    #     return encoding["input_ids"], encoding["attention_mask"]

    # def tok_decode(self, tokens, skip_special_tokens=True):
    #     return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_generate(self, inputs, stop, **gen_kwargs):
        # TODO: handle max_length
        # gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 1024
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        # if "top_p" not in gen_kwargs:
        #     gen_kwargs["top_p"] = None
        # if "num_beams" not in gen_kwargs:
        #     gen_kwargs["num_beams"] = 1

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer,
            stop,
            inputs["input_ids"].shape[1],
            inputs["input_ids"].shape[0],
        )
        return self.model.generate(
            **inputs,
            # max_length=max_length,
            stopping_criteria=stopping_criteria,
            do_sample=True if gen_kwargs["temperature"] > 0 else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError(
            "model type `hf-multimodal` does not support loglikelihood_rolling. Use 'hf' model type for text-only loglikelihood_rolling tasks"
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError(
            "model type `hf-multimodal` does not support loglikelihood or multiple choice. Use 'hf' model type for text-only loglikelihood tasks"
        )

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
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
            desc="Running generate_until requests with text+image input",
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
            contexts, all_gen_kwargs, doc_to_visual, doc = zip(
                *chunk
            )  # TODO: can we cut down further on number of distinct things we pass around?

            visuals = [
                vis(d) for vis, d in zip(doc_to_visual, doc)
            ]  # TODO: I think *fully* flattening is just wrong for bs>1 ?

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

            ### end stuff that's entirely copied verbatim from HFLM ###

            max_ctx_len = self.max_length - max_gen_toks  # noqa: F841 # TODO: this assumes we are using a causal LM. is that always valid? shouldn't be

            # if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
            print(f"Prompt:\n\n{contexts}\n")

            self.tokenizer.padding_side = "left"
            inputs = self.processor(
                images=visuals, text=contexts, return_tensors="pt", padding=True
            ).to(
                self._device, self.model.dtype
            )  # TODO: factor out into a tok_batch_encode bit ; truncate from left using max_ctx_len

            context_enc = inputs["input_ids"]

            if "max_length" not in kwargs:
                kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

            cont = self._model_generate(inputs, stop=until, **gen_kwargs)

            ### essentially same as HFLM beyond this line!

            cont_toks_list = cont.tolist()
            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                # if self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM: # TODO: ensure this holds for VLMs
                cont_toks = cont_toks[context_enc.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                # if self.accelerator.is_main_process and doc_id[0] % 100 == 0:
                print(f"Generated text:\n\n{s}\n")

                res.append(s)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
