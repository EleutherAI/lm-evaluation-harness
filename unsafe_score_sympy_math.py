import argparse
import logging
import json
import textwrap
from lm_eval.utils import SymbolicMathMixin, MajorityVotingMixin, timeout
from lm_eval.evaluator import make_table
from tqdm import tqdm

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
        
        accs = []
        pass_rates = []
        for doc in tqdm(docs):
            answer = checker.parse_tex(checker.normalize_tex(doc['answer']))
            
            programs = doc['metadata']['extracted_programs']

            exec_globals = {}

            if isinstance(programs, str):
                print("Detected maj@1")

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

            else:
                raise ValueError(f"Key extracted_programs has incorrect type: {type(programs)}")


            
        results[task] = {"acc": sum(accs)/len(accs), "pass_rate": sum(pass_rates)/len(pass_rates)}

    output['results'] = results

    print(make_table(output))

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    logging.critical(
            "THIS PROGRAM EXECUTES UNTRUSTED MODEL GENERATED CODE."
            "NO EFFORT HAS BEEN TAKEN TO AVOID OS AND NETWORK SIDE EFFECTS."
            "USE WITH CAUTION."
    )

    parser = argparse.ArgumentParser("Unsafe script for scoring the sympy_math tasks")

    parser.add_argument("--output", type=str, help="path to output file from running sympy math tasks")

    args = parser.parse_args()
    main(args)

