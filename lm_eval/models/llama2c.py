import ctypes
import numpy as np
import os
from tqdm import tqdm

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

import torch
import torch.nn.functional as F

import copy
import logging
import transformers
from transformers import LlamaConfig


from lm_eval.models.utils import (
    _add_special_kwargs,
    configure_pad_token,
    handle_stop_sequences,
    has_bos_prefix,
    postprocess_generated_text,
)

from collections.abc import Iterator

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

eval_logger = logging.getLogger(__name__)


@register_model("llama2c")
class LLAMA2C(LM):
    def __init__(
        self, 
        pretrained: str | transformers.PreTrainedModel = None,
        lib_path=None, 
        ckpt_path=None, 
        tokenizer: str = "", 
        max_length: int | None = None,
        # backend: Literal["default", "causal", "seq2seq"] = "default",
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: str | None = "main",
        subfolder: str = "",
        truncation: bool | None = False,
        # logits_cache: bool = True,
        # device: str | None = "cuda",
        # dtype: str | torch.dtype | None = "auto",
        # softmax_dtype: str | torch.dtype | None = None,
        # mixed_precision_dtype: str | torch.dtype | None = None,
        # batch_size: int | str | None = 1,
        # max_batch_size: int | None = 64,
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        add_bos_token: bool | None = None,
        prefix_token_id: int | None = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # parallelize: bool | None = False,
        # max_memory_per_gpu: int | str | None = None,
        # max_cpu_memory: int | str | None = None,
        # offload_folder: str | os.PathLike | None = "./offload",
        # PEFT, delta weights and quantization options
        # peft: str | None = None,
        # delta: str | None = None,
        # autogptq: bool | str | None = False,
        # gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        # end token for thinking, either the string or int token id.
        # splits to get response after this token (if provided).
        # think_end_token: str | int | None = None,
        # enable_thinking: bool | None = None,
        # chat_template_args: dict[str, Any] | None = None,
        **kwargs,

        ):
        super().__init__()
        assert lib_path and ckpt_path, "need lib_path, ckpt_path"

        self.lib = ctypes.CDLL(lib_path)

        # --- bind C API ---
        self.lib.llama2c_create.restype = ctypes.c_void_p
        self.lib.llama2c_create.argtypes = [ctypes.c_char_p]

        self.lib.llama2c_destroy.restype = None
        self.lib.llama2c_destroy.argtypes = [ctypes.c_void_p]

        self.lib.llama2c_vocab_size.restype = ctypes.c_int
        self.lib.llama2c_vocab_size.argtypes = [ctypes.c_void_p]

        self.lib.llama2c_forward_logits.restype = ctypes.c_int
        self.lib.llama2c_forward_logits.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]

        self.h = self.lib.llama2c_create(ckpt_path.encode("utf-8"))
        self.V = int(self.lib.llama2c_vocab_size(self.h))

        revision = str(revision)

        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            subfolder=subfolder,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )

        # Tinyllama V.10 3T config
        self._config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=1,
            # dtype="float32",
            eos_token_id=2,
            head_dim=64,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=5632,
            max_position_embeddings=2048,
            mlp_bias=False,
            # model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=22,
            num_key_value_heads=4,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling=None,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            # transformers_version="4.57.3",
            use_cache=True,
            vocab_size=32000
        )

        self.truncation = truncation
        # self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)
        self.add_bos_token = add_bos_token
        self._max_length = max_length
        self.revision = revision

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )


    def __del__(self):
        try:
            if getattr(self, "h", None):
                self.lib.llama2c_destroy(self.h)
                self.h = None
        except Exception:
            pass

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def eot_token_id(self) -> int:
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        return 2048

    @property
    def max_gen_toks(self) -> int:
        return 256

    def _create_tokenizer(
        self,
        pretrained: str | transformers.PreTrainedModel,
        tokenizer: str
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | None,
        revision: str | None = "main",
        trust_remote_code: bool | None = False,
        use_fast_tokenizer: bool | None = True,
        gguf_file: str | None = None,
        add_bos_token: bool | None = None,
        subfolder: str | None = "",
    ) -> None:
        """Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if not tokenizer and gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token is not None:
            kwargs["add_bos_token"] = add_bos_token

        if subfolder:
            kwargs["subfolder"] = subfolder

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer,
                    (
                        transformers.PreTrainedTokenizer,
                        transformers.PreTrainedTokenizerFast,
                    ),
                )
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **kwargs,
    ) -> list[int]:
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = _add_special_kwargs(
            add_special_tokens, self.add_bos_token
        )
        # set add_special_tokens=False if the string already starts with BOS token.
        if add_special_tokens is None and has_bos_prefix(
            string, self.tokenizer.decode(self.prefix_token_id)
        ):
            special_tokens_kwargs["add_special_tokens"] = False
        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            original_lengths = len(encoding)
            if original_lengths > left_truncate_len:
                eval_logger.warning(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_decode(self, tokens: Iterator[list[str]], skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _forward_logits(self, token_id: int, pos: int, out: np.ndarray):
        # out: float32[V]
        self.lib.llama2c_forward_logits(
            self.h, int(token_id), int(pos),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def _softmax(self,logits: np.ndarray):
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        return ex / (float(np.sum(ex)))
    
    def _log_softmax(self,logits: np.ndarray):
        return np.log(self._softmax(logits))

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                continuation_enc = self.tok_encode(
                    continuation, add_special_tokens=False
                )
                # BOS or EOS as context: handle when context is empty -> (context + continuation) -> (BOS + continuation
                context_enc, continuation_enc = (
                    ([self.prefix_token_id], continuation_enc)
                    if self.prefix_token_id != continuation_enc[0]
                    else (continuation_enc[:1], continuation_enc[1:])
                )
                # BOS or EOS as context
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
    ) -> list[tuple[float, bool]]:
        """
        requests: List[((context_str, continuation_str), context_ids, continuation_ids)]
        return:   List[(sum_logprob, is_greedy)] in ORIGINAL request order
        """

        # Output in original order
        out: list[tuple[float, bool] | None] = [None] * len(requests)

        # --- helper: build truncated combined tokens exactly like HF logic ---
        # HF idea: keep last (max_length + 1) tokens of (context+continuation).
        # We feed combined[:-1] as inputs; at each step logits predict the next token.
        def _build_combined_trunc(ctx_ids: list[int], cont_ids: list[int]) -> list[int]:
            combined = ctx_ids + cont_ids
            if len(combined) > self.max_length + 1:
                # same warning spirit as HF
                eval_logger.warning(
                    f"Combined length (context {len(ctx_ids)} + continuation {len(cont_ids)}) "
                    f"exceeds max_length+1 ({self.max_length + 1}). Truncating from the left."
                )
            return combined[-(self.max_length + 1):]

        # --- Group only the "one-token continuation" cases by their input-token prefix ---
        # Key = tuple(inputs) where inputs = combined_trunc[:-1]
        one_tok_groups: dict[tuple[int, ...], list[tuple[int, int]]] = {}

        # Preprocess: decide which are one-token and compute their group keys
        for i, (_txt, ctx_ids, cont_ids) in enumerate(requests):
            assert len(ctx_ids) > 0
            assert len(cont_ids) > 0
            assert len(cont_ids) <= self.max_length

            combined_trunc = _build_combined_trunc(ctx_ids, cont_ids)
            inputs = combined_trunc[:-1]  # tokens we actually feed
            if len(cont_ids) == 1:
                key = tuple(inputs)
                one_tok_groups.setdefault(key, []).append((i, cont_ids[0]))

        # Progress bar over ALL requests
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )

        buf = np.empty((self.V,), dtype=np.float32)

        for key_inputs, members in one_tok_groups.items():

            pos = 0
            # feed all input tokens; after feeding the last input token,
            # buf contains logits for the ONE continuation token (the last token of combined_trunc)
            for tid in key_inputs:
                self._forward_logits(int(tid), pos, buf)
                pos += 1

            x = torch.from_numpy(buf)
            with torch.inference_mode():
                lps = F.log_softmax(x, dim=0)

            # lps = self._log_softmax(buf)
            for req_idx, target_tid in members:
                lp = lps[target_tid]
                greedy = (int(np.argmax(buf)) == int(target_tid))
                out[req_idx] = (float(lp), bool(greedy))
                pbar.update(1)

        for i, (_txt, ctx_ids, cont_ids) in enumerate(requests):
            if out[i] is not None:
                continue  # already done in one-token group

            combined_trunc = _build_combined_trunc(ctx_ids, cont_ids)

            # continuation start index within combined_trunc
            cont_start = len(combined_trunc) - len(cont_ids)

            total_lp = 0.0
            is_greedy = True

            pos = 0
            # We will feed combined_trunc[:-1] and at each step score the "next token" if it belongs to continuation.
            # After feeding token at index j, buf predicts token at index j+1.
            for j, tid in enumerate(combined_trunc[:-1]):
                self._forward_logits(int(tid), pos, buf)
                pos += 1

                next_index = j + 1
                next_tid = combined_trunc[next_index]

                if next_index >= cont_start:
                    x = torch.from_numpy(buf)
                    with torch.inference_mode():
                        lps = F.log_softmax(x, dim=0)
                    total_lp += lps[next_tid]
                    if int(np.argmax(buf)) != int(next_tid):
                        is_greedy = False

            out[i] = (float(total_lp), bool(is_greedy))
            pbar.update(1)

        pbar.close()

        return [x for x in out]


    def generate_until(self, requests, disable_tqdm: bool = False):
        res = []
        buf = np.empty((self.V,), dtype=np.float32)

        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        for (context, kwargs) in tqdm([r.args for r in requests], disable=disable_tqdm):

            if isinstance(kwargs, dict):
                kwargs = copy.deepcopy(kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise TypeError(
                    f"Expected `kwargs` to be of type `dict` but got {type(kwargs)}"
                )
            
            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            max_ctx_len = self.max_length - max_gen_toks
            assert max_ctx_len > 0, (
                f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
            )

            context_enc = self.tok_encode(
                context,
                left_truncate_len=max_ctx_len
            )

            pos = 0

            # eat prompt
            print(f"Length of prompt: {len(context_enc)}")
            for tid in context_enc:
                if(pos % 100 == 0):
                    print(f"Feed Prompt... pos {pos}")
                self._forward_logits(tid, pos, buf)
                pos += 1

            out_ids = []
            out_text = ""
            for _ in range(max_gen_toks):
                nxt = int(np.argmax(buf))
                if(pos % 100 == 0):
                    print(f"Generate answer... pos {pos},")

                out_ids.append(nxt)

                if nxt == self.eot_token_id:
                    break
                
                out_text = self.tok_decode(out_ids)
                if any(s in out_text for s in until):
                    break

                self._forward_logits(nxt, pos, buf)
                pos += 1

            s = postprocess_generated_text(
                generation=out_text,
                stop=until,
                think_end_token= None
            )
            print(s)
            res.append(s)
        return res

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError(
            "loglikelihood_rolling not yet supported for llama2c models"
        )

