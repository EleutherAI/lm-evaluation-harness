import copy
import json
import torch
import torch.nn.functional as F
from lm_eval.base import LM
from lm_eval import utils
from tqdm import tqdm
from unittest.mock import patch


class MegatronDSLM(LM):

    def __init__(self, checkpoint_path, batch_size=1, args=None, args_overrides=None,
                 world_size=1, rank=0, local_rank=0):
        """Megatron-DeepSpeed LM.

        :param checkpoint_path: str
            Path to the checkpoint folder.
        :param batch_size: int
            Batch size for inference.
        :param args: obj, optional
            Namespace object containing arguments for both megatron_args and deepspeed_args
            If not provided, we will load directly from the checkpoint
        :param args_overrides: str or dict
            Either a dictionary of override, or a path to a JSON file containing overrides
        :param world_size: int
            World size for MP
        :param rank: int
            World rank for MP
        :param local_rank: int
            Local rank
        """
        super().__init__()
        import deepspeed
        import megatron.enums
        from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
        from megatron.testing_utils import mockenv_context
        from megatron import get_tokenizer, initialize_megatron
        from megatron.checkpointing import load_checkpoint
        from megatron.training import setup_model_and_optimizer
        from pretrain_gpt import model_provider as gpt_model_provider

        # Import args
        if args is None:
            ds_checkpoint = DeepSpeedCheckpoint(
                checkpoint_path,
                tp_degree=1,
                pp_degree=1,
            )
            args = ds_checkpoint.get_args()
        self.args = copy.deepcopy(args)
        self.checkpoint_path = checkpoint_path

        # Fixed overrides
        self.args.world_size = world_size
        self.args.data_parallel_size = 1
        self.args.tensorboard_dir = None

        # We will override the load path later on, because the initializer also tries to load the
        # optimizer and scheduler, which we don't need and has additional checks which we will fail
        self.args.load = None

        # Override args
        if args_overrides is None:
            args_overrides = {}
        elif isinstance(args_overrides, str):
            args_overrides = read_json(args_overrides)
        for k, v in args_overrides.items():
            setattr(self.args, k, v)

        # Need to map some values to Enums
        if isinstance(self.args.position_embedding_type, str):
            self.args.position_embedding_type = megatron.enums.PositionEmbeddingType[
                self.args.position_embedding_type]

        # Environmental variables for distributed running
        distributed_env_variables = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "1123",
            "RANK": str(rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
        }

        # We need to hijack the _GLOBAL_ARGS variable as the Megatron/DS code will try to read from args
        # We also need to stop the _parse_args function from doing anything, as it will either try to manually
        # parse sys.argv (which does not have the information we're trying to inject), or throw an error
        # saying that the args have already initialized.
        with patch("megatron.global_vars._GLOBAL_ARGS", self.args):
            with mockenv_context(**distributed_env_variables):
                with patch('megatron.global_vars._parse_args') as parse_args_func:
                    parse_args_func.return_value = self.args
                    deepspeed.init_distributed()
                    initialize_megatron()
                    self.megatron_tokenizer = get_tokenizer()
                    megatron_ds_model, _, _ = setup_model_and_optimizer(gpt_model_provider)
                    # Override load path here
                    self.args.load = checkpoint_path
                    load_checkpoint(megatron_ds_model, None, None)

                    # Jason: I don't really know what else is in the model usually,
                    #        when I checked it was just this one element
                    megatron_ds_model = megatron_ds_model[0]
                    megatron_ds_model.module.activation_checkpoint_interval = 0
                    megatron_ds_model._compute_loss = False
                    megatron_ds_model.fwd_outputs = []
                    megatron_ds_model.eval()
                    self.megatron_ds_model = megatron_ds_model

        # Used properties
        self.EOT_TOKEN_ID = self.megatron_tokenizer.eod
        self.max_length = args.seq_length
        self.batch_size = batch_size
        self.device = torch.device("cpu")

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config={}):
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            # TODO: automatic (variable) batch size detection for vectorization
            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps = []
                contlens = []
                inplens = []

                padding_length = None

                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying

                for _, context_enc, continuation_enc in chunk:
                    # sanity check
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= self.max_length

                    # how this all works:
                    #          CTX      CONT
                    # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                    # gpt2    \               \
                    # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                    # cont_toks      4 5 6 7 8 9

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                multi_logits = F.log_softmax(self._model_call(torch.cat(inps, dim=0)),
                                             dim=-1).cpu()  # [batch, seq, vocab]

                for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens,
                                                                             contlens):
                    contlen = len(cont_toks)

                    logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                    greedy_tokens = logits.argmax(dim=-1)

                    # cont_toks :: [1, seq]
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)

                    max_equal = (greedy_tokens == cont_toks).all()

                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    answer = (float(logits.sum()), bool(max_equal))

                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

        return reord.get_original(res)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        from megatron import mpu
        from megatron.utils import get_ltor_masks_and_position_ids
        with patch("megatron.global_vars._GLOBAL_ARGS", self.args):
            # Logic taken from pretrain_gpt.get_batch_pipe
            data_b = mpu.broadcast_data(["text"], {"text": inps}, torch.int64)
            tokens = data_b['text'].long()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                self.megatron_tokenizer.eod,
                reset_position_ids=self.args.reset_position_ids,
                reset_attention_mask=self.args.reset_attention_mask,
                eod_mask_loss=self.args.eod_mask_loss,
                prefix_indices=None,
                loss_on_targets_only=self.args.loss_on_targets_only,
            )
            self.args.attn_mask = attention_mask

            # Provide Inputs
            self.megatron_ds_model.pipe_buffers["inputs"] = [(tokens, position_ids, attention_mask)]
            self.megatron_ds_model.pipe_buffers["outputs"] = [None]

            # Run model
            with torch.no_grad():
                self.megatron_ds_model._exec_forward_pass(buffer_id=0)

            # Extract output
            # batch_size, seq, vocab
            megatron_ds_output = self.megatron_ds_model.pipe_buffers["outputs"][0]

            # Prevent model from saving any state, to prevent OOM
            self.megatron_ds_model.loss = None
            self.megatron_ds_model.total_loss = None
            self.megatron_ds_model.fwd_outputs = []
            self.megatron_ds_model.pipe_buffers["outputs"] = [None]
        return megatron_ds_output[:, :, :self.megatron_tokenizer.vocab_size]

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.megatron_tokenizer.tokenizer, GPT2Tokenizer):
            return self.megatron_tokenizer.tokenizer.encode(text)
        else:
            return self.megatron_tokenizer.tokenizer.encode(text, add_special_tokens=False)

    def greedy_until(self, requests):
        raise NotImplementedError()


def read_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())
