import torch
from tqdm import tqdm
from lm_eval import utils
from lm_eval.models.huggingface import AutoCausalLM, AutoSeq2SeqLM


class GloballyNormalizedCausalLM(AutoCausalLM):
    """Support for globally normalized models.
    Identical to AutoCausalLM except there is no softmax operation during loglikelihood computation.
    """
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        return _loglikelihood_tokens_no_softmax(self, requests, disable_tqdm=disable_tqdm, override_bs=override_bs)


class GloballyNormalizedSeq2SeqLM(AutoSeq2SeqLM):
    """Support for globally normalized models.
    Identical to AutoSeq2SeqLM except there is no softmax operation during loglikelihood computation.
    """
    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        return _loglikelihood_tokens_no_softmax(self, requests, disable_tqdm=disable_tqdm, override_bs=override_bs)


# The following is a modified copy of _loglikelihood_tokens from base.py that does not use softmax (@denizyuret 20231202)

def _loglikelihood_tokens_no_softmax(_self, requests, disable_tqdm=False, override_bs=None):
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

    reordered_requests = re_ord.get_reordered()
    n_reordered_requests = len(reordered_requests)

    # automatic (variable) batch size detection for vectorization
    # pull longest context sample from request
    def _batch_scheduler(pos):
        sched = pos // int(n_reordered_requests / _self.batch_schedule)
        if sched in _self.batch_sizes:
            return _self.batch_sizes[sched]
        print(
            f"Passed argument batch_size = auto:{_self.batch_schedule}. Detecting largest batch size"
        )
        _self.batch_sizes[sched] = _self._detect_batch_size(reordered_requests, pos)
        print(f"Determined largest batch size: {_self.batch_sizes[sched]}")
        return _self.batch_sizes[sched]

    for chunk in utils.chunks(
        tqdm(reordered_requests, disable=disable_tqdm),
        n=_self.batch_size
        if _self.batch_size != "auto"
        else override_bs
        if override_bs is not None
        else 0,
        fn=_batch_scheduler
        if _self.batch_size == "auto"
        and n_reordered_requests > 0
        and not override_bs
        else None,
    ):
        inps = []
        cont_toks_list = []
        inplens = []

        padding_length = None

        # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
        # tensors, then we pack them together into a batch, call the model, and then pick it all apart
        # again because vectorizing is annoying

        for _, context_enc, continuation_enc in chunk:
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= _self.max_length

            # how this all works:
            #          CTX      CONT
            # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
            # gpt2    \               \
            # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
            # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :_self.vocab_size] slice

            # when too long to fit in context, truncate from the left
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(_self.max_length + 1) :][:-1],
                dtype=torch.long,
            ).to(_self.device)
            (inplen,) = inp.shape

            cont = continuation_enc

            # since in _collate we make sure length is descending, the longest is always the first one.
            padding_length = (
                padding_length if padding_length is not None else inplen
            )

            # pad length from seq to padding_length
            inp = torch.cat(
                [
                    inp,  # [seq]
                    torch.zeros(padding_length - inplen, dtype=torch.long).to(
                        inp.device
                    ),  # [padding_length - seq]
                ],
                dim=0,
            )

            inps.append(inp.unsqueeze(0))  # [1, padding_length]
            cont_toks_list.append(cont)
            inplens.append(inplen)

        batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
        # For globally normalized models we do not take softmax at each position: the sequence score is computed by summing the unnormalized token scores.
        multi_logits = _self._model_call(batched_inps).cpu()
        # multi_logits = F.log_softmax(
        #     self._model_call(batched_inps), dim=-1
        # ).cpu()  # [batch, padding_length, vocab]

        for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
            chunk, multi_logits, inps, inplens, cont_toks_list
        ):

            # Slice to original seq length
            contlen = len(cont_toks)
            inplen = inplen + (
                logits.shape[0] - padding_length
            )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
            logits = logits[inplen - contlen : inplen].unsqueeze(
                0
            )  # [1, seq, vocab]

            # Check if per-token argmax is exactly equal to continuation
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(
                0
            )  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()

            # Obtain log-probs at the corresponding continuation token indices
            # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
            logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                -1
            )  # [1, seq]

            # Answer: (log prob, is-exact-match)
            answer = (float(logits.sum()), bool(max_equal))

            # partial caching
            if cache_key is not None:
                _self.cache_hook.add_partial("loglikelihood", cache_key, answer)

            res.append(answer)

    return re_ord.get_original(res)


