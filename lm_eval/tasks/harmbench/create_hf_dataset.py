"""
Create the Hugging Face dataset for the HarmBench benchmark.

This script expects the following files to be in the same directory:
- human_jailbreak_test.json
- human_jailbreak_val.json
"""

import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

ROOT_DIR = os.path.dirname((os.path.abspath(__file__)))


# --- Log into Hugging Face ---
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
assert hf_token, "HF_TOKEN not found in .env"

login(token=hf_token)
api = HfApi()
repo_id = "swiss-ai/harmbench"


# --- Load data ---
def load_csv_url_as_df(url):
    full_url = (
        "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/"
        + url
    )
    return pd.read_csv(full_url)


def load_local_json_as_df(json_path):
    with open(os.path.join(ROOT_DIR, json_path), "r") as f:
        data = json.load(f)

    # Expand the data so each behavior in the array becomes a separate row
    expanded_data = []
    for behavior_id, behaviors in data.items():
        # Each key contains an array of 5 behaviors
        for i, behavior in enumerate(behaviors):
            # Create a unique ID for each behavior: original_id + index
            unique_id = f"{behavior_id}_{i+1}"
            expanded_data.append({"BehaviorID": unique_id, "Behavior": behavior})

    return pd.DataFrame(expanded_data)


direct_val = load_csv_url_as_df("harmbench_behaviors_text_val.csv")
direct_test = load_csv_url_as_df("harmbench_behaviors_text_test.csv")
metadata = load_csv_url_as_df("harmbench_behaviors_text_all.csv")
human_test = load_local_json_as_df("human_jailbreak_test.json")
human_val = load_local_json_as_df("human_jailbreak_val.json")


# --- Create DatasetDicts ---
direct_request = DatasetDict(
    {
        "val": Dataset.from_pandas(direct_val),
        "test": Dataset.from_pandas(direct_test),
    }
)

human_jailbreaks = DatasetDict(
    {
        "val": Dataset.from_pandas(human_val),
        "test": Dataset.from_pandas(human_test),
    }
)

# --- Push datasets ---
direct_request.push_to_hub(repo_id, config_name="DirectRequest", private=True)
human_jailbreaks.push_to_hub(repo_id, config_name="HumanJailbreaks", private=True)

metadata.to_csv(os.path.join(ROOT_DIR, "metadata.csv"), index=False)
api.upload_file(
    path_or_fileobj=os.path.join(ROOT_DIR, "metadata.csv"),
    path_in_repo="metadata.csv",
    repo_id=repo_id,
    repo_type="dataset",
)
