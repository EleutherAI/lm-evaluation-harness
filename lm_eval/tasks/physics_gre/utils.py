import datasets
from sklearn.metrics import accuracy_score


def process_docs(dataset: datasets.Dataset):
    # Remove the data points that have images in the input. 100 -> 76
    dataset = dataset.filter(lambda x: x["has_image"] is False)
    # Remove the data points without groud truth label. 76 -> 75
    dataset = dataset.filter(lambda x: x["target_scores"] is not None)
    # All questions must have one and only one correct answer.
    assert (
        len(dataset.filter(lambda x: sum(x["target_scores"].values()) != 1)) == 0
    ), "Zero or More than one correct answers."

    def _helper(doc):
        label = next(key for key, value in doc["target_scores"].items() if value == 1)
        doc["label"] = ["A", "B", "C", "D", "E"].index(label)
        return doc

    return dataset.map(_helper)


def raw_score(references, predictions):
    return accuracy_score(references, predictions)


def percentile(references, predictions):
    return accuracy_score(references, predictions)


def raw_score_agg(items):
    score_agg = sum(items) / len(items)
    # For the Physics GRE, each correct answer is worth 1 point
    #   and each incorrect answer results in a -0.25 reduction.
    score_agg = max(0, score_agg - 0.25 * (1 - score_agg))
    score_agg = round(score_agg * 100)
    return int(score_agg)


def percentile_agg(items):
    score_agg = raw_score_agg(items)
    return raw_score_to_percentile(score_agg)


def raw_score_to_percentile(score_agg):
    score_percentile_mapping = {
        range(81, 101): 98,
        range(77, 81): 97,
        range(75, 77): 96,
        range(72, 75): 95,
        71: 94,
        range(69, 71): 93,
        range(67, 69): 92,
        range(65, 67): 91,
        64: 90,
        63: 89,
        range(61, 63): 87,
        60: 86,
        59: 85,
        range(57, 59): 84,
        56: 82,
        55: 80,
        range(53, 55): 78,
        52: 77,
        51: 75,
        range(49, 51): 72,
        48: 70,
        47: 69,
        range(45, 47): 66,
        44: 64,
        43: 62,
        range(41, 43): 59,
        40: 57,
        39: 54,
        range(37, 39): 52,
        36: 48,
        35: 46,
        range(33, 35): 43,
        32: 41,
        range(30, 32): 38,
        29: 35,
        28: 32,
        range(26, 28): 30,
        25: 27,
        24: 25,
        range(22, 24): 22,
        21: 20,
        20: 18,
        range(18, 20): 16,
        17: 14,
        16: 12,
        range(14, 16): 10,
        13: 9,
        12: 8,
        range(10, 12): 6,
        9: 5,
        8: 4,
        range(6, 8): 3,
        5: 2,
        range(1, 5): 1,
        0: 0,
    }

    for key, value in score_percentile_mapping.items():
        if isinstance(key, int) and score_agg == key:
            return value
        elif isinstance(key, range) and score_agg in key:
            return value

    raise ValueError("Invalid Raw Score {}".format(score_agg))
