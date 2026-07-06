def filter_multiple_choice(dataset):
    """Filter out examples that have no multiple-choice targets.

    Some BigBench tasks (e.g. kanji_ascii) contain a mix of multiple-choice
    and free-form examples in the same split.  The multiple-choice task
    variant crashes when it encounters rows with an empty
    ``multiple_choice_targets`` list, so we drop them here.
    """
    return dataset.filter(lambda doc: len(doc["multiple_choice_targets"]) > 0)
