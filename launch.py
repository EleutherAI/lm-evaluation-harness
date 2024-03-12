import os
import sys
from typing import List, Optional

from lm_eval.__main__ import cli_evaluate


from datetime import datetime
import os
import importlib.util

import click
from tqdm import tqdm


MAX_WORKERS_PER_GPU = 1


def execute_config(
    model: str,
    task: str,
    batch_size: int,
    limit: int,
    output_dir: str
):
    # Save the original standard output
    import subprocess

    output_dir = os.path.join(output_dir, model, task)

    args = [
        "lm_eval",
        "--model", "based_lm",
        "--model_args", f"checkpoint_name={model}",
        "--tasks", task,
        "--device", "cuda:0",
        "--batch_size", str(batch_size),
        "--log_samples",
        "--output_path", output_dir
    ]

    if limit is not None:
        args.extend(["--limit", str(limit)])
    
    subprocess.run(args)



@click.command()
@click.option("-m", "--model", type=str, multiple=True)
@click.option("-t", "--task", type=str, multiple=True)
@click.option("-p", "--parallelize", is_flag=True)
@click.option("--gpus", default=None, type=str)
@click.option("--batch-size", default=8, type=int)
@click.option("--limit", default=None, type=int)
def main(
    model: List[str], 
    task: List[str], 
    batch_size: int,
    limit: Optional[int],
    parallelize: bool, 
    gpus: str
):

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Load the given Python file as a module
    configs = [
        {"model": m, "task": t} for m in model for t in task
    ]

    use_ray = parallelize and len(configs) > 0
    if use_ray:
        import ray
        # ray was killing workers due to OOM, but it didn't seem to be necessary 
        os.environ["RAY_memory_monitor_refresh_ms"] = "0"
        ray.init(ignore_reinit_error=True, log_to_driver=True)

    print(f"Running sweep with {len(configs)} configs")

    output_dir = f"output/{datetime.now().strftime('%y-%m-%d_%H-%M')}"

    # Run each script in parallel using Ray
    if not use_ray:
        for config in configs: 
            execute_config(
                **config,
                batch_size=batch_size,
                limit=limit,
                output_dir=output_dir
            )
    else:
        completed = 0
        total = len(configs)
        print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        remote = ray.remote(num_gpus=(1 // MAX_WORKERS_PER_GPU))(execute_config)
        futures = [remote.remote(**config, batch_size=batch_size, limit=limit, output_dir=output_dir) for config in configs]
        
        while futures:
            complete, futures = ray.wait(futures)
            completed += len(complete)
            print(f"Completed: {completed} ({completed / total:0.1%}) | Total: {total}")

        ray.shutdown()



if __name__ == "__main__":
    main()
