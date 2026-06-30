def process_label(dataset):
    def _map(doc):
        # entailment -> index 1 (True), not_entailment -> index 0 (False)
        label_map = {"not_entailment": 0, "entailment": 1}
        return {**doc, "label": label_map[doc["label"]]}
    return dataset.map(_map)