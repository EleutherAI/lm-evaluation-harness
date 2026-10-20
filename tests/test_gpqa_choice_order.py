"""GPQA choice order follows the random seed instead of a stale map cache."""

import importlib.util
import random
from pathlib import Path

from datasets import Dataset, load_from_disk


def _load_process_docs(relative: str):
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(relative.replace("/", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_docs


zeroshot = _load_process_docs("lm_eval/tasks/gpqa/zeroshot/utils.py")
cot_zeroshot = _load_process_docs("lm_eval/tasks/gpqa/cot_zeroshot/utils.py")


def _dataset(path: Path) -> Dataset:
    rows = {
        "Incorrect Answer 1": [f"wrong-a-{i}" for i in range(8)],
        "Incorrect Answer 2": [f"wrong-b-{i}" for i in range(8)],
        "Incorrect Answer 3": [f"wrong-c-{i}" for i in range(8)],
        "Correct Answer": [f"right-{i}" for i in range(8)],
    }
    dataset = Dataset.from_dict(rows)
    dataset.save_to_disk(path)
    return load_from_disk(path)


def _letters(dataset: Dataset) -> list[str]:
    return list(dataset["answer"])


def test_cached_process_docs_follows_the_new_seed(tmp_path):
    source = _dataset(tmp_path / "gpqa")
    Dataset.from_dict(
        {column: source[column] for column in source.column_names}
    ).save_to_disk(tmp_path / "gpqa-fresh")

    random.seed(1)
    first = _letters(zeroshot(load_from_disk(tmp_path / "gpqa")))

    random.seed(1234)
    warmed = _letters(zeroshot(load_from_disk(tmp_path / "gpqa")))

    random.seed(1234)
    fresh = _letters(zeroshot(load_from_disk(tmp_path / "gpqa-fresh")))

    assert warmed == fresh
    assert first != fresh
    for row in zeroshot(load_from_disk(tmp_path / "gpqa")):
        letter = row["answer"]
        column = {"(A)": "choice1", "(B)": "choice2", "(C)": "choice3", "(D)": "choice4"}[
            letter
        ]
        assert row[column].startswith("right-")


def test_cold_order_matches_one_sequential_shuffle(tmp_path):
    source = _dataset(tmp_path / "gpqa")
    random.seed(7)
    state = random.getstate()
    processed = zeroshot(source)

    rng = random.Random()
    rng.setstate(state)
    expected = []
    for correct in source["Correct Answer"]:
        choices = [f"ignored-{i}" for i in range(3)] + [correct]
        # The real choices differ per row, but the permutation depends only on length.
        rng.shuffle(choices)
        expected.append(choices.index(correct))

    actual = [
        {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3}[letter]
        for letter in processed["answer"]
    ]
    assert actual == expected
    # The chain-of-thought variant uses the same shuffle.
    random.seed(7)
    assert _letters(cot_zeroshot(load_from_disk(tmp_path / "gpqa"))) == _letters(
        processed
    )
