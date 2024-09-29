import copy
import logging
from collections import defaultdict
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from packaging import version
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import StoppingCriteriaList

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import stop_sequences_criteria


try:
    NEURON_AVAILABLE = True
    from optimum.neuron import NeuronModelForCausalLM
    from optimum.neuron.generation import TokenSelector
    from optimum.neuron.version import __version__ as optimum_neuron_version
except ImportError:
    NeuronModelForCausalLM = object
    NEURON_AVAILABLE = False


logger = logging.getLogger(__name__)


class CustomNeuronModelForCausalLM(NeuronModelForCausalLM):
    """NeuronModelForCausalLM with `stopping_criteria` in `generate`"""

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        generation_config: Optional["GenerationConfig"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

        This method uses the same logits processors/warpers and stopping criteria as the transformers library
        `generate()` method but restricts the generation to greedy search and sampling.

        It does not support transformers `generate()` advanced options.

        Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        for details on generation configuration.

        Parameters:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~transformers.generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.

        Returns:
            `torch.Tensor`: A  `torch.FloatTensor`.
        """
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(
            self.generation_config if generation_config is None else generation_config
        )
        model_kwargs = generation_config.update(
            **kwargs
        )  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = TokenSelector.create(
            input_ids, generation_config, self, self.max_length
        )
        selector.stopping_criteria.append(stopping_criteria)
        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size and not self.continuous_batching:
            logger.warning(
                "Inputs will be padded to match the model static batch size. This will increase latency."
            )
            padding_shape = [self.batch_size - batch_size, sequence_length]
            padding = torch.full(
                padding_shape, fill_value=self.config.eos_token_id, dtype=torch.int64
            )
            padded_input_ids = torch.cat([input_ids, padding])
            if attention_mask is not None:
                padding = torch.zeros(padding_shape, dtype=torch.int64)
                padded_attention_mask = torch.cat([attention_mask, padding])

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            attention_mask=padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]


@register_model("neuronx")
class NEURON_HF(TemplateLM):
    """
    Enables usage with on AWS Neuron
    using the HuggingFace Transformers + Transformers neuronx library.
    Tested with neuron 2.17.0
    """

    def __init__(
        self,
        pretrained: Optional[str] = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        revision: Optional[str] = "main",
        tp_degree: Optional[int] = None,
        subfolder: Optional[str] = None,
        tokenizer: Optional[str] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[int] = 1,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        if not NEURON_AVAILABLE:
            raise Exception(
                "Tried to load neuron model, but neuron is not installed ",
                "please install neuron via pip install transformers-neuron ",
                "also make sure you are running on an AWS inf2 instance",
            )
        if version.parse(optimum_neuron_version) != version.parse("0.0.24"):
            logger.warning(
                '`optimum-neuron` model requires `pip install "optimum[neuronx]>=0.0.17" '
                "preferably using the Hugging Face Neuron Deep Learning AMI (Ubuntu 22.04) "
                "https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2 "
                f"You are using optimum-neuron={optimum_neuron_version}"
            )
        super().__init__()

        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = int(batch_size)
        batch_size = int(batch_size)

        self._config = transformers.AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )

        revision = str(revision)  # cast to string if not already one
        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_tokenizer,
        )

        neuron_config = getattr(self._config, "neuron", None)
        if neuron_config is None:
            # Check export parameters
            if tp_degree is not None:
                assert isinstance(tp_degree, int), (
                    f"tp_degree must be set to an integer,"
                    f" but is tp_degree=`{tp_degree}` with type=`{type(tp_degree)}`."
                    "Set it to a number lower than the number of neuron cores on your instance."
                    " For inf2.xlarge and inf2.8xlarge, set it to `2`."
                    " For inf2.24xlarge, set it <= `12`."
                    " For inf2.48xlarge, set it <= `24`."
                )
            torch_dtype = lm_eval.models.utils.get_dtype(dtype)

            if torch_dtype == torch.float16:
                self.amp_dtype = "f16"
            elif torch_dtype == torch.bfloat16:
                self.amp_dtype = "bf16"
            elif torch_dtype == torch.float32:
                self.amp_dtype = "f32"
            else:
                raise NotImplementedError(
                    "Only float16/bfloat16/float32 are supported."
                )

            print(f"{'='*20} \n exporting model to neuron")
            self.model = CustomNeuronModelForCausalLM.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
                export=True,
                batch_size=batch_size,
                num_cores=tp_degree,
                auto_cast_type=self.amp_dtype,
                sequence_length=max_length,
            )
            neuron_config = self.model.config.neuron
            print(
                f"SUCCESS: neuron model exported with config {neuron_config}. \n {'='*20}"
            )
        else:
            print(
                f"{'='*20} \n loading neuron model with config" f" {neuron_config}..."
            )
            self.model = CustomNeuronModelForCausalLM.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )
            print(f"SUCCESS: neuron model loaded. \n {'='*20}")

        self.truncation = truncation

        self.vocab_size = self.tokenizer.vocab_size
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.add_bos_token = add_bos_token

        self.batch_schedule = 1
        self.batch_sizes = {}

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        return self.tokenizer.bos_token_id or self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        """device are neuron cores, but the created tensors are on CPU."""
        return "cpu"

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None):
        """ """
        if add_special_tokens is None:
            add_special_tokens = False or self.add_bos_token

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ):
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = False or self.add_bos_token

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )
        if left_truncate_len:
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.

        with torch.inference_mode():
            if "do_sample" not in generation_kwargs.keys():
                generation_kwargs["do_sample"] = False

            stopping_criteria = stop_sequences_criteria(
                self.tokenizer,
                stop + [self.tokenizer.decode([self.config.eos_token_id])],
                1,
                context.shape[0],
            )

            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.eot_token_id,
                use_cache=True,
                **generation_kwargs,
            )

    def _select_cont_toks(self, logits, contlen=None, inplen=None):
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]

        return logits

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        loglikelihoods = []

        adaptive_batch_size = None

        for (string,) in tqdm(
            [req.args for req in requests], disable=(disable_tqdm or (self.rank != 0))
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            pad_amnt = 0
            if self.world_size > 1:
                # We pad out the external document-level iterator so the inner iterator doesn't hang
                mytensor = torch.tensor(len(rolling_token_windows), device=self.device)
                gathered = (
                    self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
                )

                pad_amnt = max(gathered) - gathered[self.rank]
                if pad_amnt > 0:
                    rolling_token_windows += pad_amnt * [rolling_token_windows[0]]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
                override_bs=adaptive_batch_size,
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)
            # cache this loglikelihood_rolling request
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), string_nll)
        return loglikelihoods

    def _loglikelihood_tokens(
        self, requests, disable_tqdm: bool = False, override_bs=None
    ):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)

        n_reordered_requests = len(re_ord.get_reordered())  # noqa
        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request

        chunks = lm_eval.models.utils.chunks(
            re_ord.get_reordered(),
            n=self.batch_size,
            fn=None,
        )

        for chunk in tqdm(chunks, disable=(disable_tqdm or (self.rank != 0))):
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []  # noqa
            encoder_attns = []  # noqa

            padding_len_inp = None
            padding_len_cont = None  # noqa
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape

                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # Add dummy inputs up to the model static batch size
            if len(inps) < self.batch_size:
                inps = inps + [
                    torch.zeros_like(inps[0]),
                ] * (self.batch_size - len(inps))

            masks = [torch.ones_like(inp) for inp in inps]
            batched_inps = lm_eval.models.utils.pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]

            batched_masks = lm_eval.models.utils.pad_and_concat(
                padding_len_inp, masks, padding_side="right"
            )
            if self.model.model.neuron_config.output_all_logits:
                inputs = self.model.prepare_inputs_for_prefill(
                    batched_inps, batched_masks
                )
                multi_logits = F.log_softmax(
                    self.model.forward(**inputs).logits, dim=-1
                )  # [batch, padding_length (inp or cont), vocab]
            else:
                # The model will only return the logits for the last input token, so we need
                # to iterate over inputs to accumulate logits.
                # To speed things up we use the KV cache as we would do when generating.
                inputs = self.model.prepare_inputs_for_prefill(
                    batched_inps[:, :1], batched_masks[:, :1]
                )
                outputs = [self.model.forward(**inputs).logits]
                for i in range(1, padding_len_inp):
                    inputs = self.model.prepare_inputs_for_decode(
                        batched_inps[:, : i + 1], batched_masks[:, : i + 1]
                    )
                    outputs.append(self.model.forward(**inputs).logits)
                multi_logits = F.log_softmax(torch.concat(outputs, dim=1), dim=-1)

            for (cache_key, _, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp)
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(
                    cont_toks, dtype=torch.long, device=self.device
                ).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

                if cache_key is not None:
                    # special case: loglikelihood_rolling produces a number of loglikelihood requests
                    # all with cache key None. instead do add_partial on the per-example level
                    # in the loglikelihood_rolling() function for those.
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

        return re_ord.get_original(res)

    def generate_until(self, requests, disable_tqdm: bool = False):
        res = defaultdict(list)
        re_ords = {}

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        grouper = lm_eval.models.utils.Grouper(requests, lambda x: str(x.args[1]))
        for key, reqs in grouper.get_grouped().items():
            # within each set of reqs for given kwargs, we reorder by token length, descending.
            re_ords[key] = utils.Reorderer([req.args for req in reqs], _collate)

        pbar = tqdm(total=len(requests), disable=(disable_tqdm or (self.rank != 0)))

        # for each different set of kwargs, we execute all requests, by batch.
        for key, re_ord in re_ords.items():
            chunks = lm_eval.models.utils.chunks(
                re_ord.get_reordered(), n=self.batch_size
            )
            for chunk in tqdm(chunks, disable=self.rank != 0):
                contexts, all_gen_kwargs = zip(*chunk)
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
                        f"Expected `kwargs` to be of type `dict` but got {kwargs}"
                    )
                # add EOS token to stop sequences
                eos = self.tok_decode(self.eot_token_id)
                if not until:
                    until = [eos]
                else:
                    until.append(eos)
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks
                # first stop sequence is used to halt generation upon encountering
                primary_until = [until[0]]

                max_ctx_len = self.max_length - max_gen_toks

                # encode, pad, and truncate contexts for this batch
                context_enc, attn_masks = self.tok_batch_encode(
                    contexts,
                    left_truncate_len=max_ctx_len,
                    truncation=self.truncation,
                )
                context_enc = context_enc.to(self.device)
                attn_masks = attn_masks.to(self.device)

                if "max_length" not in kwargs:
                    kwargs["max_length"] = context_enc.shape[1] + max_gen_toks

                # perform batched generation
                cont = self._model_generate(
                    context=context_enc,
                    attention_mask=attn_masks,
                    stop=primary_until,
                    **kwargs,
                )

                cont_toks_list = cont.tolist()
                for cont_toks, context in zip(cont_toks_list, contexts):
                    # discard context + left-padding toks if using causal decoder-only LM
                    cont_toks = cont_toks[context_enc.shape[1] :]

                    s = self.tok_decode(cont_toks)

                    # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                    for term in until:
                        if len(term) > 0:
                            # ignore '' separator,
                            # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                            s = s.split(term)[0]

                    res[key].append(s)

                    self.cache_hook.add_partial(
                        "generate_until", (context, gen_kwargs), s
                    )
                    pbar.update(1)
            # reorder this group of results back to original unsorted form
            res[key] = re_ord.get_original(res[key])

        pbar.close()

        return grouper.get_original(res)
