import torch
import transformers

from tqdm import tqdm

import torch.nn.functional as F

from lm_eval import utils
from lm_eval.logger import eval_logger
from lm_eval.api.model import LM, register_model

from accelerate import Accelerator
from typing import List


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
        return self._DEFAULT_MAX_LENGTH #TODO: Is this a good default?
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
    
    def _model_call(self, inps, attn_mask = None ,labels = None):
        """
        inps: a torch tensor of shape [batch, sequence_ctx]
        the size of sequence may vary from call to call

        labels: a torch tensor of shape [batch, sequence_cont]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(input_ids = inps, attention_mask = attn_mask, labels = labels).logits
        
    def _model_generate(self, context, max_length, stop):
        
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, 1, context.shape[0]
        )

        return self.model.generate(
            context,
            max_new_tokens=max_length,
            stopping_criteria=stopping_criteria,
            do_sample=False,

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
        for (string,) in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=None,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )
            
            rolling_token_windows = [(self.eot_token_id,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(
                rolling_token_windows,
                disable_tqdm=True,
            )

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

                max_batch_length_inp = max(max_batch_length_inp, inplen) if max_batch_length_inp is not None else inplen
                max_batch_length_cont = max(max_batch_length_cont, contlen) if max_batch_length_cont is not None else contlen

                inps.append(inp)  # [1, inp_len]
                conts.append(cont) # [1, cont_len]
                encoder_attns.append(torch.ones_like(inp))

                cont_toks_list.append(continuation_enc)

            batched_inps = utils.pad_and_concat(max_batch_length_inp, inps) # [batch, padding_length]
            batched_conts = utils.pad_and_concat(max_batch_length_cont, conts) # [batch, padding_length]
            batched_encoder_mask = utils.pad_and_concat(max_batch_length_inp, encoder_attns)
            # need to make attention mask here too

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, attn_mask = batched_encoder_mask, labels = batched_conts), dim=-1
            ).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, cont_toks in zip(
                chunk, multi_logits, cont_toks_list
            ):

                # Slice to original seq length 
                contlen = len(cont_toks)
                logits = logits[: contlen].unsqueeze(
                    0
                )  # [1, seq, vocab]

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

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]
            (primary_until) = until[0]

            context_enc = torch.tensor(
                [self.tok_encode(context)[-self.max_length :]]
            ).to(self.device)

            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until
            )
            s = self.tok_decode(cont[0].tolist())
            for term in until:
                s = s.split(term)[0]

            res.append(s)

        return re_ord.get_original(res)
    


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :][
            :, -self.sequence_id_len :
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )