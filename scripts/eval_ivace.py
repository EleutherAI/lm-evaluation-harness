import os
import json
import subprocess
import argparse
import time

import yaml
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import create_repo, delete_repo
from huggingface_hub.utils import HfHubHTTPError


def is_evaluated(model_name, dataset_results):
    try:
        create_repo(dataset_results, repo_type="dataset", private=True)
        return False
    except HfHubHTTPError:
        dataset = load_dataset(dataset_results, split="train", token=True, download_mode="force_redownload")
        return model_name in dataset["model_name"]


def extract_score_and_create_summary(ds_input, task_list, model_name, model_type, user_data, results_data, score_name="f1,none"):
    model_name_dir = model_name.replace("/", "__")
    directory = f"results/{model_name_dir}"

    new_row = {
        "url": f"https://huggingface.co/{model_name}",
        "type": model_type,
        "model_name": model_name
    }

    # get json paths
    json_path_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json")]
    if not json_path_list:
        raise FileNotFoundError(f"No JSON files found in the directory {directory}.")
    
    for json_path in json_path_list:
        # load the JSON data
        with open(json_path, "r") as file:
            data = json.load(file)

        # add task metric
        task = list(data["results"].keys())[0]
        new_row[task] = data["results"][task][score_name]

        print(f"Reading task={task} from: {json_path}")

    # load existing dataset from the Hugging Face Hub
    try:
        create_repo(results_data, repo_type="dataset", private=True)
        print(f"Creating repository {results_data}")
        updated_data = pd.DataFrame([new_row])
    except HfHubHTTPError:
        print(f"Updating repository {results_data}")
        dataset = load_dataset(results_data, split="train", token=True, download_mode="force_redownload")
        if len(dataset) == 0:
            updated_data = pd.DataFrame([new_row])
        else:
            # append the new row to the dataset
            updated_data = dataset.to_pandas().reset_index(drop=True)
            updated_data = pd.concat([updated_data, pd.DataFrame([new_row])], ignore_index=True)

    # push the updated dataset back to the Hub
    updated_dataset = Dataset.from_pandas(updated_data)
    print(updated_dataset.to_pandas())
    updated_dataset.push_to_hub(results_data, private=True)
    print(f"1. Successfully updated the dataset on the Hugging Face Hub: {results_data}")

    # remove request from input user data
    print(ds_input.to_pandas())
    delete_repo(repo_id=user_data, repo_type="dataset")
    time.sleep(1)
    create_repo(user_data, repo_type="dataset", private=True)
    ds_input.push_to_hub(user_data, private=True)
    print(f"Successfully updated the user request data on the Hugging Face Hub: {user_data}")


def main(ix: int, batch_size: int, output_path: str):
    user_data = "iberbench/ivace-user-request"
    results_data = "iberbench/lm-eval-results-ac"

    # read user request dataset
    ds_input = load_dataset(user_data, split="train", token=True, download_mode="force_redownload")
    record_info = ds_input[ix]

    # read ivace tasks
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "config", "tasks.yml")
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    task_list = data.get("tasks", [])

    # check if the model is already evaluated
    if is_evaluated(record_info["model_name"], results_data):
        raise ValueError(f"{record_info['model_name']} was already evaluated")

    # filter input_dataset
    ds_input = ds_input.filter(
        lambda example, index: index != ix,
        with_indices=True
    )

    for task in task_list:
        print(f"Init task={task}...")

        # create model args
        model_args = f"pretrained={record_info['model_name']}"
        if record_info["precision_option"] == "GPTQ":
            model_args = f"{model_args},autogptq=True"
        else:
            model_args = f"{model_args},dtype={record_info['precision_option']}"

        if record_info["weight_type_option"] != "Original":
            model_args = f"{model_args},peft={record_info['base_model_name']}"

        # add rest of the args
        command = ["lm_eval", "--model", "hf", "--model_args", model_args, "--tasks", task, "--batch_size", f"{batch_size}", "--output_path", output_path]
        result = subprocess.run(command, capture_output=True, text=True)

        # print the output
        print("Output:", result.stdout)
        print("Error:", result.stderr)
        print("Return code:", result.returncode)

    # write results in a dataset
    print("persisting results in the hub")
    extract_score_and_create_summary(ds_input, task_list, record_info["model_name"], record_info["model_type"], user_data, results_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script example.")
    
    # adding arguments
    parser.add_argument("--id", type=int, required=True, help="Request number from iberbench/ivace-user-request.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size used in lm_eval")
    parser.add_argument("--output_path", type=str, default="results", help="Path to store results of lm_eval")

    # run process
    args = parser.parse_args()
    main(args.id, args.batch_size, args.output_path)
