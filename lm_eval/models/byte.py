from subprocess import Popen, PIPE, STDOUT

import torch

from lm_eval.base import BaseLM


class SubprocessLM(BaseLM):
    def __init__(self, process: Popen, max_length: int, eot_token_id: int):
        super().__init__()
        self._eot_token_id = eot_token_id
        self._max_length = max_length
        self.process = process

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, 256] with the
        logits returned from the model
        """
        prefix_len = inps.size(-1)
        batch = inps.size(0)
        outs = (
            torch.zeros_like(inps, dtype=torch.float32)
            .unsqueeze(-1)
            .expand(batch, prefix_len, 256)
            .contiguous()
        )
        for tok_i in range(prefix_len):
            batch_of_tokens = bytes(inps[:, tok_i].tolist())
            self.process.stdin.write(batch_of_tokens)
            self.process.stdin.flush()

        # float32 logits
        logits = self.process.stdout.read(batch * prefix_len * 256 * 4)
        logits = torch.frombuffer(logits, dtype=torch.float32)
        outs[:, :, :] = logits.view(prefix_len, batch, 256).permute(1, 0, 2)

        return outs

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()

    @staticmethod
    def start(cmd, max_length: int, eot_token_id: int = 0):
        process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, text=False)
        return SubprocessLM(process, max_length=max_length, eot_token_id=eot_token_id)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--max-len", required=True, type=int)
        parser.add_argument("--eot-token-id", default=0, type=int)
        parser.add_argument("cmd", nargs=argparse.REMAINDER)
        args = parser.parse_args(arg_string.split())
        return cls.start(args.cmd, max_length=args.max_len, eot_token_id=args.eot_token_id)

    @property
    def batch_size(self):
        return 1

    def tok_encode(self, string: str):
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens):
        return tokens.decode("utf-8")

    @property
    def eot_token_id(self):
        return self._eot_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 1024 * 1024 * 1024

    @property
    def device(self):
        return torch.device("cpu")
