import json
import argparse
import lm_eval.tasks
import lm_eval.models
from lm_eval.evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--description_path', default=None)
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    task_names = ['hellaswag', 'copa']
    task_dict = lm_eval.tasks.get_task_dict(task_names)
    lm = lm_eval.models.get_model('dummy')()

    description_dict = {}
    if args.description_path:
        with open(args.description_path, 'r') as f:
            description_dict = json.load(f)

    num_fewshot = args.num_fewshot
    results = evaluate(
        lm,
        task_dict,
        num_fewshot,
        args.limit,
        description_dict
    )


if __name__ == '__main__':
    main()
