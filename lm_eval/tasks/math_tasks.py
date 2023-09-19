import re
import math
import code
import signal
from abc import ABC, abstractmethod

import inspect
import lm_eval.datasets.hendrycks_math.hendrycks_math
from lm_eval.metrics import mean
from lm_eval.base import Task, rf
from lm_eval.utils import MajorityVotingMixin, SymbolicMathMixin

class SymbolicMathTask(Task, SymbolicMathMixin, MajorityVotingMixin, ABC):
    """
    Abstract base class representing tasks whose answer is a LaTeX mathematical expression,
    e.g `347`, `\sqrt{3}/3`, or `x^2 + 7`.

    Note that this template inherits from `SymbolicMathMixin`. Therefore you do not need 
    to write code for TeX expression normalization, parsing, and equivalence judgement. 

    See lm_eval.tasks.minerva_math for a reference implementation.

    Abstract properties:
        has_training_docs (bool): whether task has a training set
        has_validation_docs (bool): whether the task has a validation set
        has_test_docs (bool): whether the task has a test set. 
        training_docs (Iterator[Dict]): Iterator containing the training set. 
            Each row *must* have a field named `"answer"` that contains a LaTeX expression string.
        validation_docs (Iterator[Dict]): Iterator containing the validation set. 
            Each row *must* have a field named `"answer"` that contains a LaTeX expression string.
        test_docs (Iterator[Dict]): Iterator containing the test set. 
            Each row *must* have a field named `"answer"` that contains a LaTeX expression string.
        fewshot_context (str): context that the model will be conditioned on. Should include few shot examples
            and the test example.
        end_seq (str): when sampling from a model, stops sampling when `end_seq` is generated.
        get_unnormalized_answer (str | Literal[self.INVALID_ANSWER]): extracts a TeX expression 
            from a model sample. Returns `self.INVALID_ANSWER` if extraction failed.
    """
    MAJORITY_VOTING = "majority_voting"
    SAMPLING_TEMPERATURE = "sampling_temperature"
    TOP_P = "top_p"
    EVAL_BATCH_SIZE = "eval_batch_size"
    INVALID_ANSWER="[invalidanswer]"

    @abstractmethod
    def has_training_docs(self) -> bool:
        pass

    @abstractmethod
    def has_validation_docs(self) -> bool:
        pass

    @abstractmethod
    def has_test_docs(self) -> bool:
        pass

    @abstractmethod
    def training_docs(self):
        pass

    @abstractmethod
    def validation_docs(self):
        pass

    @abstractmethod
    def test_docs(self):
        pass

    @abstractmethod
    def fewshot_context(self, doc):
        """
        Arguments:
            doc (Dict): a dataset row. 
        Returns:
            out (str): Fewshot context corresponding to `doc`. 
        """
        pass

    @property
    @abstractmethod
    def end_seq(self) -> str:
        """
        str: Model will generate until `self.end_seq`.
        """
        pass

    @abstractmethod
    def get_unnormalized_answer(self, text: str) -> str:
        """
        Arguments:
            text (str): model sample
        Returns:
            out (str | Literal[self.INVALID_ANSWER]): string containing a TeX Expression or 
                `self.INVALID_ANSWER`. 
        """
        pass

    def doc_to_target(self):
        raise NotImplementedError("SymbolicMathTask has no doc_to_target method.")

    def doc_to_text(self, doc):
        raise NotImplementedError("SymbolicMathTask does not implement doc_to_text")

    def should_decontaminate(self):
        return False

    def process_results(self, doc, results, params={}):
        candidates = results[0]

        assert isinstance(params, dict)
        
        if self.MAJORITY_VOTING not in params:
            unnormalized_answer = self.get_unnormalized_answer(candidates)
            answer = self.normalize_tex(unnormalized_answer)

            if unnormalized_answer==self.INVALID_ANSWER:
                acc = 0
            elif self.is_tex_equiv(answer, doc['answer']):
                acc = 1 
            else: 
                acc = 0 

            pass_rate = acc
        else:
            answers = [
                    self.normalize_tex(self.get_unnormalized_answer(candidate))
                    for candidate in candidates
                    if self.get_unnormalized_answer(candidate) != self.INVALID_ANSWER
            ]
            
            acc, pass_rate, votes = self.majority_vote(
                    answers,
                    correct_answer=doc['answer'],
                    is_equiv=self.is_tex_equiv,
            )
            if votes:
                answer = votes[0][0]
            else: 
                answer = self.INVALID_ANSWER

        results = {
            "acc": acc,
            "pass_rate": pass_rate,
            "metadata": {
                "selected_answer": answer,
                "unprocessed_answers": candidates,
            }
        }

        if self.MAJORITY_VOTING in params:
            results["metadata"]["votes"] = votes

        return results

    def aggregation(self):
        return {"acc": mean, "pass_rate": mean}

    def higher_is_better(self):
        return {"acc": True, "pass_rate": True}
