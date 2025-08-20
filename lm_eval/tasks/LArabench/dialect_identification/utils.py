
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

# class_labels = [
#         "EG","DZ","SD","YE","SY","AE","JO","LY","PS",
#         "OM","QA","BH","MSA","SA","IQ","MA"
#     ]

count_label_map = {
    "egyptian": "EG",
    "algerian": "DZ",
    "sudanese": "SD",
    "yemeni": "YE",
    "syrian": "SY",
    "tunisian": "TN",
    "emirati": "AE",
    "jordanian": "JO",
    "libyan": "LY",
    "palestinian": "PS",
    "omani": "OM",
    "lebanese": "LB",
    "kuwaiti": "KW",
    "qatari": "QA",
    "bahrani": "BH",
    "bahraini": "BH",
    "modern standard arabic": "MSA",
    "msa": "MSA",
    "saudi": "SA",
    "iraqi": "IQ",
    "moroccan": "MA",
    }

def post_process(doc, results):
    gold = doc["label"]
    text = results[0].strip()

    # Build regex pattern from keys in count_label_map
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in count_label_map.keys()) + r")\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    if match:
        matched_key = match.group(0)
        # Normalize key capitalization for dictionary lookup
        label = count_label_map.get(matched_key, count_label_map.get(matched_key.lower(), "None"))
    else:
        label = "None"

    return {"eval": (label, gold)}



def evaluate(items):
    predictions, references = zip(*items)
    acc = accuracy_score(references, predictions)
    precision = precision_score(references, predictions, average="macro")
    recall = recall_score(references, predictions, average="macro")
    f1 = f1_score(references, predictions, average="macro")
    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return results