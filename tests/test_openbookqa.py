import datasets

from lm_eval.tasks import TaskManager


def test_openbookqa_fewshot_pool_excludes_held_out_demonstrations(monkeypatch):
    def row(id_, stem, choices, answer_key="A"):
        return {
            "id": id_,
            "question_stem": stem,
            "choices": {"label": ["A", "B"], "text": choices},
            "answerKey": answer_key,
        }

    train = [
        row("train-test", "  TEST   stem", ["right", "wrong"]),
        row("train-validation", "Validation stem", ["no", "yes"], "B"),
        row(
            "different-distractor", "Test stem", ["different distractor", "right"], "B"
        ),
        row("different-answer", "Test stem", ["different answer", "wrong"]),
    ]
    validation = [row("validation", "Validation stem", ["yes", "no"])]
    test = [row("test", "Test stem", ["wrong", "RIGHT"], "B")]
    published = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train),
            "validation": datasets.Dataset.from_list(validation),
            "test": datasets.Dataset.from_list(test),
        }
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda **kwargs: published.copy())

    task = TaskManager().load("openbookqa")["tasks"]["openbookqa"]
    assert list(task.training_docs()) == train
    assert list(task.validation_docs()) == validation
    assert list(task.eval_docs) == test
    assert list(task.fewshot_docs()) == train[3:]

    task.set_fewshot_seed(1)
    assert task.fewshot_context(doc=test[0], num_fewshot=1) == (
        "Test stem different answer\n\nTest stem"
    )
