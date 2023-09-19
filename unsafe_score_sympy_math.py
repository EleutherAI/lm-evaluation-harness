import argparse
import logging
import json
import textwrap
from lm_eval.utils import SymbolicMathMixin, MajorityVotingMixin, timeout
from lm_eval.evaluator import make_table
from tqdm import tqdm
import copy

def wrap_code(code: str):
    """
    Wraps code in a try-except block
    """
    code_with_indent = textwrap.indent(code, ' ')
    return f"try:\n{code_with_indent}\nexcept:\n  pass"

def answer_of_program(program: str, time_limit=15):
    """
    Executes program and extracts `answer` global
    """
    exec_globals = {}
    try:
        with timeout(seconds=time_limit):
            exec(program, None, exec_globals)

        candidate = exec_globals.get('answer', None)
        return candidate
    except (SyntaxError, TimeoutError):
        return None

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
            answer = checker.parse_tex(checker.normalize_tex(doc['answer']))
            
            programs = doc['metadata']['extracted_programs']

            exec_globals = {}

            is_majority_voting = not isinstance(programs, str)

            if not is_majority_voting:
                program_with_exception = wrap_code(programs)

                candidate = answer_of_program(program_with_exception)

                if candidate is not None:
                    acc = checker.is_exp_equiv(answer, candidate)
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
                        is_equiv=checker.is_exp_equiv
                )

                if not votes:
                    acc = 0
                    pass_rate = 0

            accs.append(acc)
            pass_rates.append(pass_rate)

            output['cache'][task][i]['acc'] = acc
            output['cache'][task][i]['pass_rate'] = pass_rate

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

