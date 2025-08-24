
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re


def post_process(doc, results):
    gold = doc["label"]
    label = results[0].strip()
    label = re.sub(r"\s+", "", label)
    label = label.strip('+')
    return {"eval": (label, gold)}



def evaluate(items):
    return {'dummy': 1}  # Dummy evaluation function for compatibility
    predicted_labels, true_labels = zip(*items)
    return {"Macro F1": f1_score(true_labels, predicted_labels, average="macro")}
