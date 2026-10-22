"""Replay frozen answers by dataset identity without running inference."""
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.judging.core import indexed, read_jsonl


@register_model("cached-rollouts")
class CachedRollouts(LM):
    tokenizer_name = "cached-rollouts"

    def __init__(self, path, **kwargs):
        super().__init__()
        self.rows = indexed(read_jsonl(path))

    def generate_until(self, requests, disable_tqdm=False):
        answers = []
        for request in requests:
            row = self.rows[request.doc["id"]]
            if row["input_hash"] != request.doc["input_hash"]:
                raise ValueError("Cached response does not match this dataset document")
            if request.args[0] != row["prompt"]:
                raise ValueError("Cached replay requires the exact original prompt, without few-shot/template changes")
            answers.append(row["answer"])
        return answers

    def loglikelihood(self, requests):
        raise NotImplementedError("Cached rollouts support generation replay only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Cached rollouts support generation replay only")
