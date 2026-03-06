
def map_labels(doc):
    label_map = {"hate": 0, "abusive": 1, "neutral": 2}
    return label_map.get(doc["label"], 2)

def process_results(doc, results):
    prediction = results[0].strip().lower()
    gold_index = map_labels(doc)
    choices = ["hate", "abusive", "neutral"]
    gold_label = choices[gold_index]

    # Simple keyword match for accuracy
    acc = 1 if gold_label in prediction else 0
    return {"acc": acc}
