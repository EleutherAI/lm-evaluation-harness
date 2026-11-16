import argparse
import json
import os
import subprocess

import pandas as pd


def parse_args():
    """
    Parses arguments.
    Returns:
        Arguments containing the names of the prediction file and the file directory to for saving the evaluation results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath",
        type=str,
        help="path to a model output file in the lm-evaluation-harness format.",
    )
    parser.add_argument(
        "--out_fdir",
        type=str,
        help="path to an output directory for saving the results.",
    )
    args = parser.parse_args()
    return args


def read_examples(fpath: str):
    """
    Reads examples from the prediction file.
    Args:
        fpath: A path to the prediction file.
    Returns:
        Lists of the sources, targets, and predictions.
    """
    examples = pd.read_json(fpath, lines=True)
    sources, targets, predictions = [], [], []
    for i, example in examples.iterrows():
        sources.append(example["doc"]["source"])
        targets.append(example["doc"]["correction"])
        predictions.append(example["resps"][0][0].replace("\n\n", "\n"))
    return sources, targets, predictions


def save_results(fpath: str, obj: dict):
    """
    Saves the evaluation results.
    Args:
        fpath: A path for the output file for saving the results.
        obj: The evaluation results.
    """
    with open(fpath, "w+", encoding="utf-8") as out:
        json.dump(obj, out, indent=3)


def evaluate(fpath: str, out_fpath: str):
    """
    Runs the evaluation based on the ERRANT performance metric.
    Args:
        fpath: A path to the prediction file.
        out_Fpath: A path for the output file for saving the results.
    """
    tmp_name = fpath.replace(".jsonl", "").replace("/", "-")
    os.makedirs("tmp", exist_ok=True)
    sources, targets, predictions = read_examples(fpath=fpath)
    with open(f"tmp/{tmp_name}_sources.txt", "w+") as f:
        f.write("\n".join(sources))
    with open(f"tmp/{tmp_name}_targets.txt", "w+") as f:
        f.write("\n".join(targets))
    with open(f"tmp/{tmp_name}_predictions.txt", "w+") as f:
        f.write("\n".join(predictions))
    subprocess.run(
        f"errant_parallel -orig tmp/{tmp_name}_sources.txt -cor tmp/{tmp_name}_targets.txt -out tmp/{tmp_name}_targets.m2 -lev -tok",
        shell=True,
    )
    subprocess.run(
        f"errant_parallel -orig tmp/{tmp_name}_sources.txt -cor tmp/{tmp_name}_predictions.txt -out tmp/{tmp_name}_predictions.m2 -lev -tok",
        shell=True,
    )
    output = subprocess.check_output(
        f"errant_compare -ref tmp/{tmp_name}_targets.m2 -hyp tmp/{tmp_name}_predictions.m2",
        shell=True,
    )
    f_05 = float(output.decode().strip().split("\n")[-2].split()[-1].strip())
    print(f"Prediction fpath: {fpath}\n\nERRANT: {f_05}", flush=True)
    print(f"Saving to: {out_fpath}", flush=True)
    save_results(obj={"errant": f_05}, fpath=out_fpath)
    subprocess.run(f"rm tmp/{tmp_name}_*", shell=True)


def main():
    args = parse_args()
    fpath = args.fpath
    print(f"Out: {args.out_fdir}", flush=True)
    out_fpath = fpath.replace(".jsonl", "_errant.json")
    evaluate(fpath=fpath, out_fpath=out_fpath)


if __name__ == "__main__":
    print(
        "\nWARNING: make sure you have ERRANT installed to run the evaluation! Available here: https://github.com/chrisjbryant/errant\n\n",
        flush=True,
    )
    main()
