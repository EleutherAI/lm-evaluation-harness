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
import sys
import requests
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

def get_informalize_prompt():
    response = requests.get('https://raw.githubusercontent.com/zhangir-azerbayev/ProofNet/main/eval/prompts/6shot_informalize.txt')
    response.raise_for_status()
    return response.text.strip()

class ProofNetInformalizeStatements(ProofNetAutoformalizeStatements):
    VERSION = 0
    DATASET_PATH = "hoskinson-center/proofnet"

    IN_KEY = "formal_statement"
    OUT_KEY = "generated_nl_statement"
    REF_KEY = "nl_statement"
    STOP = '\n'

    prompt = get_informalize_prompt()

    def construct_requests(self, doc, ctx):
        output = rf.greedy_until(ctx, self.STOP)
        return output

    def doc_to_text(self, doc):
        formal = doc[self.IN_KEY]
        return f'Lean matlib version:\n{formal}\nTranslate the Lean mathlib version to a natural language version:'

    def fewshot_context(
        self, doc, num_fewshot, provide_description=None, rnd=None, description=None
    ):
        example = self.doc_to_text(doc)
        ctx = self.prompt + "\n\n" + example
        return ctx

    def process_results(self, doc, results, params={}):
        output = self._parse_result(results[0])
        bleu, bleu_1 = self.calc_bleu(generated=output, gold=doc[self.REF_KEY])
        results = {
            "bleu": bleu,
            "bleu-1": bleu_1,
            "gpt-eval-correct": 0.0,
            "metadata": {
                self.OUT_KEY: output
            }
        }
        # Optional GPT evaluation if enabled in the config, eg:
        #   "gpt_eval" : { "enabled": True, "settings": { "engine": "gpt-3.5-turbo", "max_tokens": 512 } }
        if 'gpt_eval' in params and params['gpt_eval']['enabled']:
            settings = params['gpt_eval'].get('settings', {})
            evaluator = GPTEvaluator(
                engine=settings.get('engine', 'gpt-3.5-turbo'),
                max_tokens=settings.get('max_tokens', 512),
            )
            gpt_eval = evaluator.evaluate(
                formal_statement=doc[self.IN_KEY],
                informal_statement_gold=doc[self.REF_KEY],
                informal_statement_generated=output
            )
            results['gpt-eval-correct'] = gpt_eval['correct']
            results['metadata']['gpt_eval'] = {
                'reason': gpt_eval['reason'],
            }

        return results

    def aggregation(self):
        return {
            "bleu": mean,
            "bleu-1": mean,
            "gpt-eval-correct": mean
        }

    def higher_is_better(self):
        return {
            "bleu": True,
            "bleu-1": True,
            "gpt-eval-correct": True
        }

    def _parse_result(self, result):
        return result


class GPTEvaluator(object):
    def __init__(self, engine='gpt-3.5-turbo', max_tokens=512):
        self.engine = engine
        self.max_tokens = max_tokens

        # Putting this here prevents import if user doesn't enable GPTEvaluator
        import openai
        self.openai = openai
        import os
        openai.api_key = os.environ['OPENAI_API_KEY']

    def eval_informalization_prompt(self, formal_statement, informal_statement_gold, informal_statement_generated):
        prompt = """Evaluate whether the natural language theorem statement is a correct informalization of the given Lean 3 formal statement. 
    To help you decide, you are given a ground-truth natural language statement. 
    Note that a statement can be a correct informalization without matching the ground-truth informal statement.
    If a candidate informalization still contains any Lean syntax, mark it incorrect.


    Format your evaluation as:
    Reason: justification for Yes/No/Unsure
    Correct: Yes/No/Unsure

    Example:
    Formal statement in Lean: theorem exercise_1_6_11 {A B : Type*} [group A] [group B] : \n  A × B ≃* B × A :=
    Ground-truth informal statement: Let $A$ and $B$ be groups. Prove that $A \\times B \\cong B \\times A$.
    Informal statement: Let $A$ and $B$ be groups. Prove that $A + B$ is isomorphic to $B × A$.

    Reason: The informal statement uses the wrong operation symbol for the product of groups.
    Correct: No


    Formal statement in Lean: %s
    Ground-truth informal statement: %s
    Informal statement: %s

    Reason:""" % (formal_statement, informal_statement_gold, informal_statement_generated)
        return prompt

    def _call_api(self, prompt, engine, max_tokens, max_retries=10, retry_wait=2):
        for i in range(max_retries):
            try:
                return self.openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant who is an expert in the Lean theorem prover."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0
                )
            except self.openai.error.OpenAIError as e:
                print("GPTEvaluator OpenAIError: %s\nRetrying (%d/%d)" % (e, i+1, max_retries))
                import time
                time.sleep(retry_wait)
        # If we exhausted the retries, return an empty message
        return {'choices': [{'message': {'content': ''}}]}

    def _parse_message(self, msg):
        parsed = {
            'correct': 0.0,
            'reason': '',
        }
        try:
            content = msg['choices'][0]['message']['content']
            correct = content.strip().split('\n')[1]
            correct_float = 1.0 if 'Yes' in correct else 0.0
            parsed['correct'] = correct_float
            parsed['reason'] = content.strip().split('\n')[0].strip().replace("Reason: ", "")
        except (IndexError, KeyError):
            # NOTE: setting correct to 0.0 here may yield a false negative.
            parsed['correct'] = 0.0
            parsed['reason'] = 'ERROR: Parsing error in GPT output (msg: %s)' % msg
        return parsed

    def evaluate(self, formal_statement, informal_statement_gold, informal_statement_generated):
        msg = self._call_api(
            prompt=self.eval_informalization_prompt(
                formal_statement=formal_statement,
                informal_statement_gold=informal_statement_gold,
                informal_statement_generated=informal_statement_generated
            ),
            engine=self.engine,
            max_tokens=self.max_tokens
        )
        evaluation = self._parse_message(msg)
        return evaluation
