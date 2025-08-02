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
repo_id = "swiss-ai/hallulens"


# --- Load data ---


def load_local_json_as_df(json_path):
    # it's a jsonl file, so we can read it line by line
    with open(json_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Convert to DataFrame
    df = pd.DataFrame(data)

    return df

precise_wiki = load_local_json_as_df("/cluster/raid/home/stea/HalluLens/data/precise_qa/prompts_precise_wiki.jsonl")
longwiki = load_local_json_as_df("/cluster/raid/home/stea/HalluLens/data/longwiki/prompts_longwiki.jsonl")
generated_entities = load_local_json_as_df("/cluster/raid/home/stea/data/auto_non_existing/save/prompts_generated_entities.jsonl")
mixed_entities = load_local_json_as_df("/cluster/raid/home/stea/data/nonsense_all/save/prompts_mixed_entities.jsonl")


# --- Create DatasetDicts ---
precise_wiki = Dataset.from_pandas(precise_wiki)
longwiki = Dataset.from_pandas(longwiki)
generated_entities = Dataset.from_pandas(generated_entities)
mixed_entities = Dataset.from_pandas(mixed_entities)

# --- Push datasets ---
def push_dataset(dataset, name):
    dataset = DatasetDict({"test": dataset})
    dataset.push_to_hub(repo_id, config_name=name, private=True)

# if any of the datasets already exist, they will be overwritten
push_dataset(precise_wiki, "precise_wiki")
push_dataset(longwiki, "longwiki")
push_dataset(generated_entities, "generated_entities")
push_dataset(mixed_entities, "mixed_entities")

# delete the train split if it exists
for dataset_name in [precise_wiki, longwiki, generated_entities, mixed_entities]:
    if "train" in dataset_name:
        del dataset_name["train"]

print("Datasets pushed successfully!")

