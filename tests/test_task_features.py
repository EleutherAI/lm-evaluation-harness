import datasets

from lm_eval.api.task import _features_from_dict


def test_infinitebench_schema():
    """The spec shipped in `_infinitebench_common_yaml`."""
    features = _features_from_dict(
        {
            "id": {"dtype": "int64"},
            "context": {"dtype": "string"},
            "input": {"dtype": "string"},
            "answer": {"dtype": "string", "sequence": True},
            "options": {"dtype": "string", "sequence": True},
        }
    )
    assert features == datasets.Features(
        {
            "id": datasets.Value("int64"),
            "context": datasets.Value("string"),
            "input": datasets.Value("string"),
            "answer": datasets.Sequence(datasets.Value("string")),
            "options": datasets.Sequence(datasets.Value("string")),
        }
    )


def test_sequence_false_stays_scalar():
    features = _features_from_dict({"answer": {"dtype": "string", "sequence": False}})
    assert features == datasets.Features({"answer": datasets.Value("string")})


def test_nested_struct():
    features = _features_from_dict(
        {"meta": {"source": {"dtype": "string"}, "page": {"dtype": "int32"}}}
    )
    assert features == datasets.Features(
        {"meta": {"source": datasets.Value("string"), "page": datasets.Value("int32")}}
    )


def test_returns_features_instance():
    """`download()` guards on this to avoid reparsing an already-converted spec."""
    assert isinstance(
        _features_from_dict({"id": {"dtype": "int64"}}), datasets.Features
    )
