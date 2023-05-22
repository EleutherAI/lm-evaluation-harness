"""
ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics
https://arxiv.org/pdf/2302.12433.pdf

The ProofNet benchmarks consists of 371 examples, each consisting of a formal
theorem statement in Lean 3, a natural language theorem statement,
and a natural language proof. The problems are primarily drawn from popular
undergraduate pure mathematics textbooks and cover topics such as real and
complex analysis, linear algebra, abstract algebra, and topology.
The ProofNet dataset supports several tasks (refer to the "Supported Tasks"
section in the paper). Of these, this evaluation harness supports:
1.  Autoformalization of statements (`proofnet_autoformalize_statements`):
    given an informal statement, produce a corresponding formal statement.
2.  Informalization of statements (`proofnet_informalize_statements`):
    given a formal statement, produce a corresponding informal statement.

Homepage: https://github.com/zhangir-azerbayev/ProofNet
"""
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from lm_eval.metrics import mean
from lm_eval.base import Task, rf

from transformers import AutoTokenizer

_CITATION = """
@misc{azerbayev2023proofnet,
      title={ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics}, 
      author={Zhangir Azerbayev and Bartosz Piotrowski and Hailey Schoelkopf and Edward W. Ayers and Dragomir Radev and Jeremy Avigad},
      year={2023},
      eprint={2302.12433},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""


class ProofNetAutoformalizeStatements(Task):
    VERSION = 0
    DATASET_PATH = "hoskinson-center/proofnet"

    BEFORE_EXAMPLE = "Natural language version: \""
    AFTER_EXAMPLE = "\"\nTranslate the natural language version to a Lean mathlib version:\n```\n"
    IN_KEY = "nl_statement"
    OUT_KEY = "gpt_formal_statement"
    REF_KEY = "formal_statement"
    STOP = "```"

    TOKENIZER = AutoTokenizer.from_pretrained('facebook/galactica-1.3b')

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        text = self.BEFORE_EXAMPLE + doc[self.IN_KEY] + self.AFTER_EXAMPLE
        return text

    def doc_to_target(self, doc):
        target = doc[self.REF_KEY] + "```"
        return target

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        ctx = super().fewshot_context(
            doc, num_fewshot, provide_description, rnd, description
        )
        assert ctx.endswith('\n')
        ctx = ctx + 'theorem'
        return ctx

    def construct_requests(self, doc, ctx):
        output = rf.greedy_until(ctx, self.STOP)
        return output

    def process_results(self, doc, results):
        output = self._parse_result(results[0])
        bleu, bleu_1 = self.calc_bleu(generated=output, gold=doc[self.REF_KEY])
        results = {
            "bleu": bleu,
            "bleu-1": bleu_1,
            "metadata": {
                self.OUT_KEY: output
            }
        }
        return results

    def calc_bleu(self, generated, gold):
        generated_tok = self.TOKENIZER.encode(generated)
        gold_tok = self.TOKENIZER.encode(gold)
        bleu = sentence_bleu(
            references=[gold_tok],
            hypothesis=generated_tok,
            smoothing_function=SmoothingFunction().method4
        )
        bleu_1 = sentence_bleu(
            references=[gold_tok],
            hypothesis=generated_tok,
            smoothing_function=SmoothingFunction().method4,
            weights=(1.0, 0.0, 0.0, 0.0)
        )
        return bleu, bleu_1

    def aggregation(self):
        return {
            "bleu": mean,
            "bleu-1": mean
        }

    def higher_is_better(self):
        return {
            "bleu": True,
            "bleu-1": True
        }

    def _parse_result(self, result):
        result = 'theorem ' + result
        return result


class ProofNetInformalizeStatements(ProofNetAutoformalizeStatements):
    VERSION = 0
    DATASET_PATH = "hoskinson-center/proofnet"

    BEFORE_EXAMPLE = "Lean mathlib version:\n"
    AFTER_EXAMPLE = '\nTranslate the Lean mathlib version to a natural language version:\n```'
    IN_KEY = "formal_statement"
    OUT_KEY = "gpt_nl_statement"
    REF_KEY = "nl_statement"
    STOP = '```'

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        ctx = Task.fewshot_context(
            self, doc, num_fewshot, provide_description, rnd, description
        )
        return ctx

    def doc_to_target(self, doc):
        target = doc[self.REF_KEY] + "```"
        return target

    def _parse_result(self, result):
        return result
