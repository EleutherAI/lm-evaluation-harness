# lm_eval/tasks/indic_sentiment_ta/utils.py
from typing import Dict
import datasets

LABELS = ["negative", "positive"]  # Tamil: 2-class only

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Normalize IndicSentiment Tamil rows for lm-eval:

    1. Filter out rows where LABEL or INDIC REVIEW is missing
       or LABEL is not in {"negative", "positive"}.
    2. Add:
        - text      : Tamil review text
        - label     : normalized string label
        - label_idx : index into LABELS
    """

    # ---- 1) FILTER: decide which rows to keep ----
    def _keep(example: Dict) -> bool:
        text = example.get("INDIC REVIEW", None)
        label_val = example.get("LABEL", None)

        if text is None or label_val is None:
            return False

        label_str = str(label_val).strip().lower()
        return label_str in LABELS

    dataset = dataset.filter(_keep)

    # ---- 2) MAP: transform kept rows (no dropping here) ----
    def _proc(example: Dict) -> Dict:
        text = example["INDIC REVIEW"]
        label_str = str(example["LABEL"]).strip().lower()

        return {
            # keep only what we need; you can also add other original fields if you like
            "text": text,
            "label": label_str,
            "label_idx": LABELS.index(label_str),
        }

    dataset = dataset.map(_proc)

    return dataset
