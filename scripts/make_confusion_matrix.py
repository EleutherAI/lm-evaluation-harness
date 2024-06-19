import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
import sklearn.metrics


allowed_metrics = ["acc", "acc_norm"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        required=True,
        metavar="DIR|DIR/file.json",
        help="Path of json file containing sample predictions.",
    )
    parser.add_argument(
        "--metric",
        default="acc",
        choices=allowed_metrics,
        help="Method of determining prediction.",
    )
    return parser.parse_args()


def confusion_matrix(file: str, metric: str):
    file = Path(file)
    assert file.exists(), f"File does not exist! ({file})"

    sample_dump = []
    with open(file, "r") as f:
        for line in f:
            sample = json.loads(line)
            sample_dump.append(sample)

    choices = [d["arg_1"].strip() for d in sample_dump[0]["arguments"].values()]
    completion_len = np.array([float(len(i)) for i in choices])
    print(f"Classes: {choices}")

    targets, predictions = [], []
    for sample in sample_dump:
        target = sample["target"]
        if isinstance(target, int):
            target_label = choices[target]
            targets.append(target_label)
        elif isinstance(target, str):
            assert target in choices
            targets.append(target)
        elif isinstance(target, list):
            raise NotImplementedError(
                "No support yet for multi-label confusion matrix!"
            )

        results = sample["resps"]
        lls = [float(r[0][0]) for r in results]
        pred = np.argmax(lls)
        pred_norm = np.argmax(lls / completion_len)
        if metric == "acc":
            pred_label = choices[pred]
        elif metric == "acc_norm":
            pred_label = choices[pred_norm]
        predictions.append(pred_label)

    cm = sklearn.metrics.confusion_matrix(targets, predictions, labels=choices)
    print(make_confusion_matrix(cm))


def make_confusion_matrix(cm):
    from pytablewriter import MarkdownTableWriter

    md_writer = MarkdownTableWriter()
    classes = [f"C{n}" for n in range(cm.shape[0])]
    # true classes in row, predicted classes in column
    md_writer.headers = ["t/p"] + classes
    values = []
    for i, c in enumerate(classes):
        values.append([c] + cm[i].tolist())
    md_writer.value_matrix = values
    return md_writer.dumps()


if __name__ == "__main__":
    args = parse_args()

    confusion_matrix(args.file, args.metric)
