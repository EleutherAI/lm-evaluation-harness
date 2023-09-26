import argparse
import logging
import json
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO


def answer_of_program(program: str):
    def _run():
        f = StringIO()
        with redirect_stdout(f):
            exec(program)
            answer = f.getvalue().strip()
            return answer
    try:
        answer = _run()
    except Exception as e:
        answer = None
    return answer


def parse_float(text):
    try:
        answer = float(text)
        return answer
    except (ValueError, TypeError):
        return None


def main(args):
    with open(args.output) as f:
        output = json.load(f)

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
        for i, doc in enumerate(tqdm(docs[:limit])):
            answer = float(doc['output_answer'])

            program = doc['metadata']['program'] + "\nprint(solution())"

            candidate = answer_of_program(program)
            candidate_answer = parse_float(candidate)
            if answer == candidate_answer:
                acc = 1
            else:
                acc = 0

            accs.append(acc)

            output['cache'][task][i]['acc'] = acc

        results[task] = {"acc": sum(accs)/len(accs)}

    output['results'] = results

    with open(args.output, 'w') as f:
        f.write(json.dumps(output, indent=4))

    print(output['results'])


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

