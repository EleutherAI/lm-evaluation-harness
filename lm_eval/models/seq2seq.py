import torch
import transformers

import copy
from tqdm import tqdm

import torch.nn.functional as F

from lm_eval import utils
from lm_eval.logger import eval_logger
from lm_eval.api.registry import register_model
from lm_eval.api.model import LM

from lm_eval.utils import MultiTokenEOSCriteria, stop_sequences_criteria

from accelerate import Accelerator


@register_model("hf-seq2seq", "seq2seq")
class Seq2SeqHFLM(LM):
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        device="cuda",
        pretrained="t5-small",
        revision="main",
        low_cpu_mem_usage=None,
        subfolder=None,
        tokenizer=None,
        batch_size=1,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)
        gpus = torch.cuda.device_count()
        if gpus <= 1:
            if device:
                if device not in ["cuda", "cpu"]:
                    device = int(device)
                self._device = torch.device(device)
                print(f"Using device '{device}'")
            else:
                print("Device not specified")
                print(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
            self._rank = 0
            self._world_size = 1

        else:
            self._device = "cpu"

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            pretrained, revision=revision, low_cpu_mem_usage=low_cpu_mem_usage
        ).to(self.device)
        self.model.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            revision=revision,
        )

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size

        if gpus > 1:
            accelerator = Accelerator()
            if gpus > accelerator.num_processes:
                warning = (
                    "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                    "If you would like to use data parallelism, please launch the script "
                    "with 'accelerate launch *script*'. "
                    f"Current run will proceed with {accelerator.num_processes} devices."
                )
                print(warning)
                self._rank = accelerator.local_process_index
                self._world_size = accelerator.num_processes
                # manually set model to use gpu, for case where many GPUs available but
                # only seek to use one
                self._device = (
                    torch.device(f"cuda:{accelerator.local_process_index}")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                self.model.to(self.device)
            else:
                self.model = accelerator.prepare(self.model)
                self._device = torch.device(f"cuda:{accelerator.local_process_index}")
                self.accelerator = accelerator

                if self.accelerator.is_local_main_process:
                    print(f"Using {gpus} devices with data parallelism")

                self._rank = self.accelerator.local_process_index
                self._world_size = self.accelerator.num_processes

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._DEFAULT_MAX_LENGTH  # TODO: Is this a good default?

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=True)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        inps: a torch tensor of shape [batch, sequence_ctx]
        the size of sequence may vary from call to call

        labels: a torch tensor of shape [batch, sequence_cont]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(
                input_ids=inps, attention_mask=attn_mask, labels=labels
            ).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # we require users to pass do_sample=True explicitly
        # for non-greedy gen. This should be reevaluated when considering beam search.
        if "do_sample" not in generation_kwargs.keys():
            generation_kwargs["do_sample"] = False
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, context.shape[0]
        )
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self.model).generate(
                context,
                max_new_tokens=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.eot_token_id,
                **generation_kwargs,
            )
        else:
            return self.model.generate(
                context,
                max_new_tokens=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.eot_token_id,
                **generation_kwargs,
            )

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # end of text as context
                context_enc = [self.eot_token_id]
            else:
                context_enc = self.tok_encode(context)

            continuation_enc = self.tok_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        loglikelihoods = []
        for (string,) in tqdm([req.args for req in requests], disable=(self.rank != 0)):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.eot_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder
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
                rolling_token_windows, disable_tqdm=True
            )

            if (self.world_size > 1) and (pad_amnt > 0):
                string_nll = [x[0] for x in string_nll[:-pad_amnt]]
            else:
                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
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
        for chunk in utils.chunks(
            tqdm(re_ord.get_reordered(), disable=(disable_tqdm or (self.rank != 0))),
            self.batch_size,
        ):
            inps = []
            conts = []
            encoder_attns = []
            cont_toks_list = []

            max_batch_length_inp = None
            max_batch_length_cont = None

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                inp = torch.tensor(
                    (context_enc)[-self.max_length :],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = torch.tensor(
                    (continuation_enc)[-self.max_length :],
                    dtype=torch.long,
                ).to(self.device)
                (contlen,) = cont.shape

                max_batch_length_inp = (
                    max(max_batch_length_inp, inplen)
                    if max_batch_length_inp is not None
                    else inplen
                )
                max_batch_length_cont = (
                    max(max_batch_length_cont, contlen)
                    if max_batch_length_cont is not None
                    else contlen
                )

                inps.append(inp)  # [1, inp_len]
                conts.append(cont)  # [1, cont_len]
                encoder_attns.append(torch.ones_like(inp))

                cont_toks_list.append(continuation_enc)

            batched_inps = utils.pad_and_concat(
                max_batch_length_inp, inps
            )  # [batch, padding_length]
            batched_conts = utils.pad_and_concat(
                max_batch_length_cont, conts
            )  # [batch, padding_length]
            batched_encoder_mask = utils.pad_and_concat(
                max_batch_length_inp, encoder_attns
            )
            # need to make attention mask here too

            multi_logits = F.log_softmax(
                self._model_call(
                    batched_inps, attn_mask=batched_encoder_mask, labels=batched_conts
                ),
                dim=-1,
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, cont_toks in zip(
                chunk, multi_logits, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                logits = logits[:contlen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                    0
                )  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                    -1
                )  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                res.append(answer)

        return re_ord.get_original(res)

    def greedy_until(self, requests):
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer([req.args for req in requests], _collate)

        for context, gen_kwargs in tqdm(re_ord.get_reordered()):
            until = None
            if isinstance(gen_kwargs, dict):
                gen_kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                print(gen_kwargs)
                if "until" in gen_kwargs.keys():
                    until = gen_kwargs.pop("until")
                    if isinstance(until, str):
                        until = [gen_kwargs]
                    elif not isinstance(until, list):
                        raise ValueError(
                            f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {until}"
                        )
            else:
                raise ValueError(
                    f"Expected `gen_kwargs` to be of type `dict` but got {gen_kwargs}"
                )
            if not until:
                until = [self.tok_decode(self.eot_token_id)]
            if "max_gen_toks" in gen_kwargs.keys():
                max_gen_toks = gen_kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self.max_gen_toks

            (primary_until) = until[0]

            context_enc = torch.tensor(
                [self.tok_encode(context)[-self.max_length :]]
            ).to(self.device)

            cont = self._model_generate(
                context=context_enc,
                max_length=context_enc.shape[1] + max_gen_toks,
                stop=primary_until,
                **gen_kwargs,
            )
            s = self.tok_decode(cont[0].tolist())
            print(s)
            for term in until:
                s = s.split(term)[0]
            print(s)
            res.append(s)

        return re_ord.get_original(res)
