"""
A utility script that pushes all Bigbench subtasks from their form in the `bigbench` HF dataset
into `{org name}/bigbench`.

Prior to running, log into HF Hub for the target HF hub org via `huggingface-cli login`.

Requires the installation of
`pip install "bigbench @ https://storage.googleapis.com/public_research_data/bigbench/bigbench-0.0.1.tar.gz"`
and is included so that the bigbench dependency can be avoided.
"""

import bigbench.api.util as bb_utils
import datasets
from tqdm import tqdm


all_task_names = bb_utils.get_all_json_task_names()

num_shots = [0]

for shots in num_shots:
    for task_name in tqdm(all_task_names):
        try:
            print(f"Loading '{task_name}' with num_shots={shots}...")
            task_ds = datasets.load_dataset("bigbench", name=task_name, num_shots=shots)

            print(f"Pushing '{task_name}' with num_shots={shots}...")
            task_ds.push_to_hub("hails/bigbench", task_name + "_zero_shot")

            del task_ds
        except Exception as e:
            raise e
