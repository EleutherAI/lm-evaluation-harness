import datasets
import pytest

from lm_eval.tasks import TaskManager


@pytest.mark.parametrize(
    ("task_name", "test_split"), [("gsm_plus", "test"), ("gsm_plus_mini", "testmini")]
)
def test_gsm_plus_fewshot_uses_gsm8k_train(monkeypatch, task_name, test_split):
    evaluation = datasets.Dataset.from_list(
        [
            {
                "question": f"Variant {i}",
                "solution": f"Evaluation solution {i}\n#### {i}",
                "answer": str(i),
                "seed_question": "Shared seed",
            }
            for i in range(6)
        ]
    )
    training_rows = [
        {"question": f"Training question {i}", "answer": f"Explanation {i}\n#### {i}"}
        for i in range(5)
    ]
    training = datasets.Dataset.from_list(training_rows)

    def load_dataset(path, name=None, *, split=None):
        if path == "qintongli/GSM-Plus":
            return datasets.DatasetDict({test_split: evaluation})
        assert (path, name, split) == ("openai/gsm8k", "main", "train")
        return training

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    task = TaskManager().load(task_name)["tasks"][task_name]
    task.set_fewshot_seed(seed=1234)
    task.build_all_requests(limit=1)
    prompt = task.instances[0].args[0]

    assert prompt.count("Question: ") == 6
    assert prompt.endswith("Question: Variant 0\nAnswer:")
    assert "Evaluation solution" not in prompt
    for row in training_rows:
        assert f"Question: {row['question']}\nAnswer: {row['answer']}" in prompt
    assert task.doc_to_target(evaluation[0]) == evaluation[0]["solution"]
