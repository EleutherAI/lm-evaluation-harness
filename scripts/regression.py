import argparse
import json
import os
import subprocess
import time


seq2seq_models = ['google/flan-t5-small']
causal_models = ['gpt2', 'facebook/opt-125m', 'EleutherAI/gpt-neo-125m', 'EleutherAI/pythia-160m']
model_names = seq2seq_models + causal_models


completion_tasks = ['boolq', 'lambada_openai', 'winogrande']
choice_tasks = ['hellaswag', 'openbookqa', 'piqa']
perplexity_tasks = ['wikitext']
generation_tasks = []
task_names = completion_tasks + choice_tasks + perplexity_tasks + generation_tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', type=str, default=None)
    parser.add_argument("--batch_size", type=str, default='auto')
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--models", default=model_names)
    parser.add_argument("--tasks", default=task_names)
    parser.add_argument("--model", type=str, default='hf-causal-experimental')
    return parser.parse_args()


def eval_models(branch, args):
    results = {}
    for model in args.models:
        # TODO: implement hf-auto to pick between causal and seq2seq models
        model_type = 'hf-causal-experimental' if model in causal_models else 'hf-seq2seq' if model in seq2seq_models else args.model
        # TODO: implement load_in_4bit from new transformers
        model_args = f'pretrained={model},use_accelerate=True,load_in_8bit=True'
        # TODO: split_and_pad_windows in AutoSeq2SeqLM doesn't exist
        tasks = args.tasks if model in causal_models or model_type == 'hf-causal-experimental' else list(filter(lambda task: task not in perplexity_tasks, args.tasks))
        # TODO: OOM with auto for seq2seq models, also can OOM with llama
        batch_size = args.batch_size if model in causal_models or model_type == 'hf-causal-experimental' else 64

        ts = time.time()

        command = f"python3 main.py --model {model_type} --model_args {model_args} --tasks {','.join(tasks)} --batch_size {batch_size} --limit {args.limit} --no_cache --output_path data/regression-{ts}.json"

        print(f'{"=" * 80}\nEvaluating {model} on {", ".join(tasks)} at {branch} with:\n\n{command}\n{"=" * 80}')

        if os.system(command) != 0:
            exit(1)

        results[model] = json.load(open(f'data/regression-{ts}.json'))

    return results


def extract_value(data, model, task, err=False):
    if model not in data:
        return 0
    results = data[model]['results']
    if task not in results:
        return 0
    results = results[task]
    if 'acc' in results:
        return results['acc'] if not err else results['acc_stderr']
    if 'word_perplexity' in results:
        return results['word_perplexity'] if not err else 0
    return 0


def main():
    args = parse_args()
    args.models = args.models.split(',') if type(args.models) == str else args.models
    args.tasks = args.tasks.split(',') if type(args.tasks) == str else args.tasks

    initial_branch = subprocess.check_output('git branch --show-current', shell=True).decode('ascii').strip()

    # TODO: implement proper timing for each task

    start1 = time.time()
    data1 = eval_models(initial_branch, args)
    end1 = time.time()

    if args.branch == None:
        start2 = 0
        data2 = {}
        end2 = 0
    else:
        if os.system(f'git checkout {args.branch}') != 0:
            exit()
        start2 = time.time()
        data2 = eval_models(args.branch, args)
        end2 = time.time()
        os.system(f'git checkout {initial_branch}')

    print(f'|model|{"|".join(map(lambda task: f"{task} {initial_branch}|{task} {args.branch}|diff", args.tasks))}|')
    print(f'|--|{"--|" * 3 * len(args.tasks)}')
    for model in args.models:
        vals = []
        for task in args.tasks:
            acc1 = 100 * extract_value(data1, model, task)
            acc2 = 100 * extract_value(data2, model, task)
            err1 = 100 * extract_value(data1, model, task, err=True)
            err2 = 100 * extract_value(data2, model, task, err=True)
            vals.append(f"{acc1:.2f}{f' ± {err1:.2f}' if err1 != 0 else ''}")
            vals.append(f"{acc2:.2f}{f' ± {err2:.2f}' if err2 != 0 else ''}")
            vals.append(f"{acc2 - acc1:.2f}")
        print(f'|{model}|{"|".join(vals)}|')
    print(f"\nTiming: {initial_branch}: {end1 - start1:.0f}s, {args.branch}: {end2 - start2:.0f}s")


if __name__ == "__main__":
    main()
