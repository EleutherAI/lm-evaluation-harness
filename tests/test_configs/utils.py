def custom_df_1(*args, **kwargs):
    from datasets import DatasetDict

    split = lambda x: [
        {
            "question": f"This is {x} question {i}",
            "target": int(i % 2 == 0),
            "choices": ["choice1", "choice2"],
        }
        for i in range(100)
    ]
    return DatasetDict(
        {
            "test": split("test"),
            "validation": split("validation"),
            "train": split("validation"),
        }
    )  # type:ignore[no-matching-overload]


def custom_df_3(*args, **kwargs):
    """Dataset with *different* target distributions per split.

    test split:       targets alternate 0, 1, 0, 1 ...  (DummyLM acc = 0.5)
    validation split: targets are always 0               (DummyLM acc = 1.0)
    """
    from datasets import DatasetDict

    return DatasetDict(
        {
            "test": [
                {
                    "question": f"This is test question {i}",
                    "target": int(i % 2 == 0),
                    "choices": ["choice1", "choice2"],
                }
                for i in range(100)
            ],
            "validation": [
                {
                    "question": f"This is validation question {i}",
                    "target": 0,
                    "choices": ["choice1", "choice2"],
                }
                for i in range(100)
            ],
            "train": [
                {
                    "question": f"This is train question {i}",
                    "target": 0,
                    "choices": ["choice1", "choice2"],
                }
                for i in range(100)
            ],
        }
    )  # type:ignore[no-matching-overload]


def custom_df_2(*args, **kwargs):
    from datasets import DatasetDict

    split = lambda x: [
        {
            "question": f"This is {x} question {i}",
            "target": 0,
            "choices": ["choice1", "choice2"],
        }
        for i in range(100)
    ]
    return DatasetDict(
        {
            "test": split("test"),
            "validation": split("validation"),
            "train": split("validation"),
        }
    )  # type:ignore[no-matching-overload]
