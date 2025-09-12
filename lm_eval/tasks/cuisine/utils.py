def doc_to_choice(doc):
    return [option[0] for option in doc["options"]]

def doc_to_target(doc):
    options = doc_to_choice(doc)
    return options.index(doc["correct_answer"][0])

def custom_exact_match(items):
    return items

def agg_custom_exact_match(items):
    options_map = {
        "أ": 0, "ا": 0, "ب": 1, "ج": 2, "د": 3, "ه": 4, "و": 4,
        "a": 0, "b": 1, "c": 2, "d": 3, "e": 4
    }
    targets, preds = zip(*items)
    targets = list(map(int, targets))
    # Convert `preds` to int (since `targets` are already integers)
    preds = [options_map.get(pred[0].lower(), pred) for pred in preds]
    return sum(t == p for t, p in zip(targets, preds)) / len(targets)
