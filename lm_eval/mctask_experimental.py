""" Multiple Choice Format Experiments.
TODO: Generalize the formatting of fewshot examples.
"""
import os
import abc
import hashlib
from argparse import ArgumentError
from dataclasses import dataclass
import typing
from attr import field
import numpy as np
import lm_eval.base as base
from lm_eval.metrics import mean


@dataclass
class MultipleChoiceDoc:
    """ Structure for storing documents. """
    question: str
    keys: typing.List[str]  # Should these be the same type as gold?
    options: typing.List[str]
    gold: int
    id: int = field(init=False)
    context: str = None  # Any extra context prior to the question.

    def __post_init__(self):
        self.id = hashlib.sha224(self.question.encode('utf-8')).hexdigest()


class BaseMultipleChoiceTask(base.Task, abc.ABC):
    """ Base Multiple Choice Task """

    def doc_to_text(self, doc: MultipleChoiceDoc):
        ctx = f"{doc.context}\n" if doc.context else ""
        return ctx + self.format_prompt(doc)

    @abc.abstractclassmethod
    def format_prompt(cls, doc: MultipleChoiceDoc) -> str:
        pass

    @abc.abstractmethod
    def doc_to_target(self, doc: MultipleChoiceDoc) -> str:
        pass

    @abc.abstractmethod
    def loglikelihood_continuation(self, doc: MultipleChoiceDoc) -> typing.List[str]:
        pass

    def construct_requests(self, doc: MultipleChoiceDoc, ctx: str):
        lls = []
        conts = self.loglikelihood_continuation(doc)
        #print(f"\n\n{conts}\n\n")
        for cont in conts:
            lls.append(base.rf.loglikelihood(ctx, f" {cont}")[0])
        return lls

    def process_results(self, doc: MultipleChoiceDoc, results: typing.List):
        gold = doc.gold
        ans = np.argmax(results)
        is_correct = 1. if ans == gold else 0.
        # Normalize by completion length.
        conts = self.loglikelihood_continuation(doc)
        completion_len = np.array([float(len(i)) for i in conts])
        acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.
        return {
            "acc": is_correct,
            "acc_norm": acc_norm,
            # Bundle answers: (model_answer, model_answer_index, is_correct, question_id).
            "answer_bundle": (doc.keys[ans], ans, is_correct, doc.id),
        }

    def higher_is_better(self):
        return {
            "acc": True,
            "acc_norm": True,
            "answer_bundle": True,
        }

    def aggregation(self):
        return {
            "acc": mean,
            "acc_norm": mean,
            "answer_bundle": answer_bundle
        }

    # UNCOMMENT TO WRITE OUT THE QUESTION TABLE
    # TODO: Write a function for this.
    #
    # def process_results(self, doc: MultipleChoiceDoc, results: typing.List):
    #     gold = doc.gold
    #     ans = np.argmax(results)
    #     is_correct = 1. if ans == gold else 0.
    #     # Normalize by completion length.
    #     conts = self.loglikelihood_continuation(doc)
    #     completion_len = np.array([float(len(i)) for i in conts])
    #     acc_norm = 1. if np.argmax(results / completion_len) == gold else 0.
    #     return {
    #         "acc": is_correct,
    #         "acc_norm": acc_norm,
    #         # Bundle questions: (question_id, question, option_0, option_1, option_2, option_3)
    #         "question_bundle": (doc.id, doc.question, doc.options),
    #     }

    # def higher_is_better(self):
    #     return {
    #         "acc": True,
    #         "acc_norm": True,
    #         "question_bundle": True,
    #     }

    # def aggregation(self):
    #     return {
    #         "acc": mean,
    #         "acc_norm": mean,
    #         "question_bundle": question_bundle,
    #     }


def answer_bundle(items):
    """ Bundles answers into a csv file. """
    from pathlib import Path
    import csv
    cols = ["model_answer", "model_answer_index", "is_correct", "question_id"]
    rows = [*items]
    path = os.environ["QUESTION_RESULT_PATH"]
    with open(f'{path}/question-by-question-results.csv', 'a', encoding="utf-8") as f:
        write = csv.writer(f)
        write.writerow(cols)
        write.writerows(rows)
    return 0


def question_bundle(items):
    """ Bundles questions into a csv file. """
    from pathlib import Path
    import csv
    options = items[0][2]
    options_name = [f"option_{i}" for i in range(len(options))]
    cols = ["question_id","question", *options_name]

    path = os.environ["QUESTION_RESULT_PATH"]
    f = open(f'{path}/question-table.csv', 'a', encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(cols)
    for item in items:
        writer.writerow([item[0],item[1],*item[2]])

    f.close()
    return 0


def key2num(doc: MultipleChoiceDoc, key: str) -> int:
    """ Maps document keys to numeric 1-based indices. """
    return str(doc.keys.index(key) + 1)  # `+ 1` for 1-based indexing.


def key2letter(doc: MultipleChoiceDoc, key: str) -> str:
    """ Maps keys to capital alphabet letters. """
    A_ascii = 65
    ascii_offset = doc.keys.index(key)
    letter = chr(A_ascii + ascii_offset)
    return letter


def format_key(key: str, type: str):
    """ Formats a multiple choice key. E.g.
        format_key("A", "period") => "A."
        format_key("A", "parens") => "(A)"
        format_key("A",  "colon") => "A:"
    Args:
    - type: "period" | "parens" | "colon"
    """
    if type == "parens":
        return f"({key})"
    elif type == "period":
        return f"{key}."
    elif type == "colon":
        return f"{key}:"
    else:
        raise ArgumentError()


class MC_NoOptionList_OptionLL_Task(BaseMultipleChoiceTask):
    """ "freeform"
    Format:
        <Context>
        Question: <question>
        Answer: 
    Continuation:
        loglikelihood_continuation = <option_i>
    """

    def format_prompt(cls, doc: MultipleChoiceDoc) -> str:
        prompt = "Question: " + doc.question + "\n"
        prompt += "Answer:"
        return prompt

    def doc_to_target(self, doc: MultipleChoiceDoc) -> str:
        return " " + doc.options[doc.gold]

    def loglikelihood_continuation(self, doc: MultipleChoiceDoc) -> typing.List[str]:
        return [option for option in doc.options]


class MC_WithOptionList_OptionLL_Task(BaseMultipleChoiceTask):
    """ "option"
    Format:
        <Context>
        Question: <question>
        <key1>: <option1>
        <key2>: <option2>
        ...
        Answer: 
    Continuation:
        loglikelihood_continuation = <option_i>
    """
    def format_prompt(cls, doc: MultipleChoiceDoc) -> str:
        prompt = "Question: " + doc.question + "\n"
        prompt += "\n".join([
            f"{format_key(doc.keys[i], 'colon')} {option}"
            for i, option in enumerate(doc.options)
        ])
        prompt += "\nAnswer:"
        return prompt

    def doc_to_target(self, doc: MultipleChoiceDoc) -> str:
        return " " + doc.options[doc.gold]

    def loglikelihood_continuation(self, doc: MultipleChoiceDoc) -> typing.List[str]:
        return [option for option in doc.options]


class MC_WithOptionList_LetterLL_Task(BaseMultipleChoiceTask):
    """ "letter"
    Format:
        <Context>
        Question: <question>
        A: <option1>
        B: <option2>
        ...
        Answer: 
    Continuation:
        loglikelihood_continuation = <key_i>
    """
    def format_prompt(cls, doc: MultipleChoiceDoc) -> str:
        prompt = "Question: " + doc.question + "\n"
        prompt += "\n".join([
            f"{format_key(key2letter(doc, doc.keys[i]), 'colon')} {option}"
            for i, option in enumerate(doc.options)
        ])
        prompt += "\nAnswer:"
        return prompt

    def doc_to_target(self, doc: MultipleChoiceDoc) -> str:
        return " " + doc.keys[doc.gold]

    def loglikelihood_continuation(self, doc: MultipleChoiceDoc) -> typing.List[str]:
        return [key for key in doc.keys]


class MC_WithOptionList_NumLL_Task(BaseMultipleChoiceTask):
    """ "number"
    Format:
        <Context>
        Question: <question>
        1: <option1>
        2: <option2>
        ...
        Answer: 
    Continuation:
        loglikelihood_continuation = <key2num(key_i)>
    """
    def format_prompt(cls, doc: MultipleChoiceDoc) -> str:
        prompt = "Question: " + doc.question + "\n"
        prompt += "\n".join([
            f"{format_key(key2num(doc, doc.keys[i]), 'colon')} {option}"
            for i, option in enumerate(doc.options)
        ])
        prompt += "\nAnswer:"
        return prompt

    def doc_to_target(self, doc: MultipleChoiceDoc) -> str:
        return f" {doc.gold + 1}"  # `+ 1` for 1-based indexing.

    def loglikelihood_continuation(self, doc: MultipleChoiceDoc) -> typing.List[str]:
        return [key2num(doc, key) for key in doc.keys]


# TODO: Try to come up with a way to do this it at runtime.
if os.environ["MC_SETTING"] == "freeform":
    MULTIPLE_CHOICE_TASK = MC_NoOptionList_OptionLL_Task
elif os.environ["MC_SETTING"] == "option":
    MULTIPLE_CHOICE_TASK = MC_WithOptionList_OptionLL_Task
elif os.environ["MC_SETTING"] == "letter":
    MULTIPLE_CHOICE_TASK = MC_WithOptionList_LetterLL_Task
elif os.environ["MC_SETTING"] == "number":
    MULTIPLE_CHOICE_TASK = MC_WithOptionList_NumLL_Task
else:
    print("No such MC_SETTING:", os.environ["MC_SETTING"])
