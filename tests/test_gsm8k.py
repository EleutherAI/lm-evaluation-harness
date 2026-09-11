from pathlib import Path

from lm_eval.api.instance import Instance
from lm_eval.api.metrics import sample_stddev
from lm_eval.api.registry import get_metric
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks._yaml_loader import load_yaml


GSM8K_CONFIG = Path(__file__).parents[1] / "lm_eval" / "tasks" / "gsm8k" / "gsm8k.yaml"


def _load_gsm8k_config():
    return load_yaml(GSM8K_CONFIG)


def _build_declared_filters(config):
    filters = {}
    for filter_config in config["filter_list"]:
        components = [
            (
                function["function"],
                {key: value for key, value in function.items() if key != "function"},
            )
            for function in filter_config["filter"]
        ]
        filters[filter_config["name"]] = build_filter_ensemble(
            filter_config["name"], components
        )
    return filters


def _apply_filter(filter_ensemble, response):
    instance = Instance(
        request_type="generate_until",
        doc={},
        arguments=(),
        idx=0,
        resps=[response],
    )
    filter_ensemble.apply([instance])
    return instance.filtered_resps[filter_ensemble.name]


def test_gsm8k_declared_extraction_filters():
    config = _load_gsm8k_config()
    filters = _build_declared_filters(config)

    assert _apply_filter(
        filters["strict-match"], "999 #### 1,234"
    ) == "1,234"

    assert _apply_filter(
        filters["flexible-extract"], "We calculate 999 first, then the answer is 1,234"
    ) == "1,234"


def test_gsm8k_declared_exact_match_normalization():
    config = _load_gsm8k_config()
    metric_config = next(
        metric for metric in config["metric_list"] if metric["metric"] == "exact_match"
    )
    metric_kwargs = {
        key: value
        for key, value in metric_config.items()
        if key not in {"metric", "aggregation", "higher_is_better", "hf_evaluate"}
    }

    metric = get_metric("exact_match")

    result = metric(
        predictions=["$1,234."],
        references=["1234"],
        **metric_kwargs,
    )

    assert result["exact_match"] == 1.0


def test_sample_stddev_exact():
    sample = [0.0, 0.5, 1.0]

    expected = 0.5

    assert sample_stddev(sample) == expected
