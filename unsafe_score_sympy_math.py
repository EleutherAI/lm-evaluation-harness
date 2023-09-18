import argparse
import logging
import json
import textwrap
from lm_eval.utils import SymbolicMathMixin, MajorityVotingMixin, timeout
from lm_eval.evaluator import make_table
from tqdm import tqdm
import copy

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

            if isinstance(programs, str):
                program_with_indent = textwrap.indent(programs, '  ')

                program_with_exception = f"try:\n{program_with_indent}\nexcept:\n  pass"

                try:
                    with timeout(seconds=15):
                        exec(program_with_exception, None, exec_globals)

                    candidate = exec_globals.get('answer', None)
                    acc = checker.is_exp_equiv(answer, candidate)
                    pass_rate = acc
                except (SyntaxError, TimeoutError):
                    acc = 0 
                    pass_rate = 0

                accs.append(acc)
                pass_rates.append(pass_rate)

                output['cache'][task][i]['acc'] = acc
                output['cache'][task][i]['pass_rate'] = pass_rate

            else:
                raise ValueError(f"Key extracted_programs has incorrect type: {type(programs)}")
    
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

