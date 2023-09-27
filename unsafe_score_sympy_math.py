import argparse
import logging
import json
import textwrap
from lm_eval.mixins import SymbolicMathMixin, MajorityVotingMixin
from lm_eval.utils import timeout
from lm_eval.evaluator import make_table
from tqdm import tqdm
import copy
from typing import Union
from numpy import isclose, isfinite
import sympy
from functools import partial
import code

def wrap_code(code: str):
    """
    Wraps code in a try-except block
    """
    code_with_indent = textwrap.indent(code, ' ')
    return f"try:\n{code_with_indent}\nexcept:\n  pass"

def cast_numeric_if_possible(x):
    try:
        with timeout(seconds=5):
            try:
                if not isfinite(int(x)):
                    raise ValueError
                return int(x)
            except (TypeError, ValueError, AttributeError):
                try:
                    if not isfinite(float(x)):
                        raise ValueError
                    return float(x)
                except (TypeError, ValueError, AttributeError):
                    return x
    except (TimeoutError, OverflowError):
        return x

def answer_of_program(program: str, time_limit=5):
    """
    Executes program and extracts `answer` global
    """
    exec_globals = {}
    try:
        with timeout(seconds=time_limit):
            exec(program, None, exec_globals)

        candidate = exec_globals.get('answer', None)

        return cast_numeric_if_possible(candidate)
    except (SyntaxError, TimeoutError):
        return None

def is_equiv(
        x1: Union[sympy.Basic, int, float], 
        x2: Union[sympy.Basic, int, float], 
        checker: MajorityVotingMixin
):
    if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
        try:
            return bool(isclose(x1, x2) or isclose(x2, x1))
        except TypeError:
            return False
    elif isinstance(x1, sympy.Basic) and isinstance(x2, sympy.Basic):
        return checker.is_exp_equiv(x1, x2)
    else:
        return False

def main(args):
    with open(args.output) as f:
        output = json.load(f)

    voter = MajorityVotingMixin()
    checker = SymbolicMathMixin()

    tasks = [task for task in output['versions']]

    results = {}
    for task in tasks:
        logging.info(f"Scoring task {task}")

        docs = output['cache'][task]

        if args.limit:
            limit = args.limit
        else:
            limit = len(docs)

        accs = []
        pass_rates = []
        for i, doc in enumerate(tqdm(docs[:limit])):
            answer = cast_numeric_if_possible(checker.parse_tex(checker.normalize_tex(doc['answer'])))
            
            programs = doc['metadata']['extracted_programs']

            exec_globals = {}

            is_majority_voting = not isinstance(programs, str)

            if not is_majority_voting:
                program_with_exception = wrap_code(programs)

                candidate = answer_of_program(program_with_exception)

                if candidate is not None:
                    acc = is_equiv(answer, candidate, checker=checker)
                    pass_rate = acc
                else:
                    acc = 0 
                    pass_rate = 0

            else:
                programs_with_exception = list(map(wrap_code, programs))
                candidates = list(map(answer_of_program, programs_with_exception))

                acc, pass_rate, votes = voter.majority_vote(
                        candidates, 
                        correct_answer=answer, 
                        is_equiv=lambda x, y: is_equiv(x, y, checker=checker)
                )

                if not votes:
                    candidate = "[invalidanswer]"
                else:
                    candidate = votes[0][0]


            accs.append(acc)
            pass_rates.append(pass_rate)

            output['cache'][task][i]['acc'] = acc
            output['cache'][task][i]['pass_rate'] = pass_rate
            output['cache'][task][i]['answer_type'] = str(type(answer))

            if is_majority_voting: 
                output['cache'][task][i]['votes'] = votes

    
        results[task] = {"acc": sum(accs)/len(accs), "pass_rate": sum(pass_rates)/len(pass_rates)}

    output['results'] = results

    with open(args.output, 'w') as f:
        f.write(json.dumps(output, indent=4))

    print(make_table(output))

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.critical(
            "THIS PROGRAM EXECUTES UNTRUSTED MODEL GENERATED CODE."
            "THERE HAS BEEN NO EFFORT TO AVOID OS AND NETWORK SIDE EFFECTS."
            "USE WITH CAUTION."
    )

    parser = argparse.ArgumentParser("Unsafe script for scoring the sympy_math tasks")

    parser.add_argument("--output", type=str, help="path to output file from running sympy math tasks")
    parser.add_argument("--limit", type=int, default=None, help="for debugging purposes, max examples per task to process")

    args = parser.parse_args()
    main(args)
