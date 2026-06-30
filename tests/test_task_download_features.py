import pytest
from datasets import Features, Value, Sequence
from lm_eval.api.task import Task
from lm_eval.config.task import TaskConfig

def test_parse_features_dict_in_download(monkeypatch):
    config = TaskConfig(task="dummy", dataset_path="dummy_path", dataset_name="dummy_name")
    
    class DummyTask(Task):
        def has_training_docs(self): return False
        def has_validation_docs(self): return False
        def has_test_docs(self): return False
        def training_docs(self): return []
        def validation_docs(self): return []
        def test_docs(self): return []
        def fewshot_docs(self): return []
        def doc_to_text(self, doc): return ""
        def doc_to_target(self, doc): return ""
        def construct_requests(self, doc, ctx, **kwargs): return []
        def process_results(self, doc, results): return {}
        def aggregation(self): return {}
        def higher_is_better(self): return {}

    task = DummyTask(config)
    
    captured_kwargs = {}
    def mock_load_dataset(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return "mock_dataset"

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", mock_load_dataset)

    dataset_kwargs = {
        "features": {
            "id": {"dtype": "int64"},
            "answer": {"dtype": "string", "sequence": True}
        }
    }

    task.download(dataset_kwargs=dataset_kwargs)

    assert "features" in captured_kwargs
    features = captured_kwargs["features"]
    
    assert isinstance(features, Features)
    assert isinstance(features["id"], Value)
    assert features["id"].dtype == "int64"
    assert isinstance(features["answer"], Sequence)
    assert isinstance(features["answer"].feature, Value)
    assert features["answer"].feature.dtype == "string"
