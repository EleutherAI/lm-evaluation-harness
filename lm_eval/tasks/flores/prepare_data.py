#!/usr/bin/env python3
"""
prepare_flores_dataset.py

Download FLORES+ devtest data for English and multiple Arabic dialects
and save it as a single CSV file.

Usage examples:
    python prepare_flores_dataset.py --hf_token <TOKEN> --output_path /path/to/benchmarks
    python prepare_flores_dataset.py --token_file creds.yaml --output_path /path/to/benchmarks
    python prepare_flores_dataset.py --output_path /path/to/benchmarks
"""

import os
import argparse
import yaml
import pandas as pd
from datasets import load_dataset


def get_hf_token(args):
    """Determine Hugging Face token priority:
    1. Direct --hf_token argument
    2. YAML file via --token_file
    3. Environment variable HF_TOKEN or HUGGINGFACE_TOKEN
    4. None (unauthenticated access)
    """
    if args.hf_token:
        print("🔹 Using Hugging Face token from command line argument.")
        return args.hf_token
    if args.token_file:
        try:
            with open(args.token_file, "r") as f:
                creds = yaml.safe_load(f)
            if isinstance(creds, dict):
                print(f"🔹 Using Hugging Face token from file: {args.token_file}")
                return creds.get("hf_token")
        except Exception as e:
            print(f"⚠️ Failed to read token from {args.token_file}: {e}")
    print("🔹 Using Hugging Face token from environment variable if not none.")
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def prepare_flores_dataset(token: str, output_path: str):
    dataset_name = "openlanguagedata/flores_plus"
    split = "devtest"

    english = "eng_Latn"
    arabic_dialects = [
        "acm_Arab",  # Mesopotamian Arabic
        "acq_Arab",  # Taʽizzi-Adeni Arabic
        "aeb_Arab",  # Tunisian Arabic
        "apc_Arab_nort3139",  # Levantine Arabic (North)
        "apc_Arab_sout3123",  # Levantine Arabic (South)
        "arb_Arab",  # Modern Standard Arabic
        "ars_Arab",  # Najdi Arabic
        "ary_Arab",  # Moroccan Arabic
        "arz_Arab",  # Egyptian Arabic
    ]

    print("🔹 Loading English dataset...")
    df = pd.DataFrame(load_dataset(dataset_name, english, split=split, token=token))
    df = df[["id", "text"]].rename(columns={"text": f"{english}_sentence"})

    for code in arabic_dialects:
        print(f"🔹 Loading {code} dataset...")
        dset = load_dataset(dataset_name, code, split=split, token=token)
        df[f"{code}_sentence"] = dset["text"]

    output_dir = os.path.join(output_path, "flores")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "devtest.csv")

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Saved FLORES+ devtest dataset to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FLORES+ devtest dataset for English and Arabic dialects.")
    parser.add_argument("--hf_token", type=str, help="Hugging Face token (optional).")
    parser.add_argument("--token_file", type=str, help="YAML file with hf_token key (optional).")
    parser.add_argument("--output_path", type=str, required=True, help="Path to benchmarks folder.")
    args = parser.parse_args()

    token = get_hf_token(args)
    prepare_flores_dataset(token, args.output_path)


if __name__ == "__main__":
    main()
