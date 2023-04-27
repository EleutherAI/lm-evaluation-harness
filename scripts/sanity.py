import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm
from lm_eval import tasks
from main import MultiChoice
from scripts.utils import rmrf_then_mkdir


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tasks", required=True, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--output_dir", default="./results/sanity_checks/", type=str)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--multiprocessing_pool_size", default=128, type=int)

    return parser.parse_args()


def sanity_check(sanity_dir, num_samples, task_name):
    """
    Sanity check on the specified task:
    - can be downloaded
    - has test or validation docs
    - dump prompts and outputs to a logfile in the specified folder
    """
    all_tasks_dict = tasks.get_task_dict(all_tasks)
    task = all_tasks_dict[task_name]

    # Dont forget to add version or the real eval will crash
    assert task.VERSION is not None

    sanity_file_path = sanity_dir / f"{task_name}.txt"

    if task.has_test_docs():
        task_doc_func = task.test_docs
    elif task.has_validation_docs():
        task_doc_func = task.validation_docs
    else:
        raise RuntimeError("Task has neither test_docs nor validation_docs")

    docs = list(task_doc_func())

    with open(sanity_file_path, "w") as f:
        for i in range(num_samples):
            doc = docs[i]
            prompt = task.doc_to_text(doc)
            target = task.doc_to_target(doc)
            s = f"{prompt}{target}\n"
            s += "-" * 80
            s += "\n"
            f.write(s)


if __name__ == "__main__":
    args = parse_args()

    all_tasks = args.tasks.split(",")
    output_dir = Path(args.output_dir)
    num_samples = args.num_samples
    pool_size = args.multiprocessing_pool_size

    rmrf_then_mkdir(output_dir)

    with Pool(pool_size) as p:
        sanity_file_list = list(
            tqdm(
                p.imap(
                    partial(sanity_check, output_dir, num_samples),
                    all_tasks,
                ),
                total=len(all_tasks),
            )
        )
