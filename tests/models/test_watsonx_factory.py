"""Offline coverage for optional configuration in the Watsonx model factory."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_eval.models import ibm_watsonx_ai as watsonx


_OMITTED = object()
_CONFIG_CASES = [
    pytest.param(_OMITTED, id="omitted"),
    pytest.param(None, id="none"),
    pytest.param({}, id="empty"),
]


@pytest.fixture
def factory_dependencies(monkeypatch):
    """Replace SDK metadata, credential lookup, and model construction only."""
    parameter_names = (
        "DECODING_METHOD",
        "LENGTH_PENALTY",
        "TEMPERATURE",
        "TOP_P",
        "TOP_K",
        "RANDOM_SEED",
        "REPETITION_PENALTY",
        "MIN_NEW_TOKENS",
        "MAX_NEW_TOKENS",
        "STOP_SEQUENCES",
        "TIME_LIMIT",
        "TRUNCATE_INPUT_TOKENS",
        "RETURN_OPTIONS",
    )
    sdk = ModuleType("ibm_watsonx_ai")
    metanames = ModuleType("ibm_watsonx_ai.metanames")
    metanames.GenTextParamsMetaNames = SimpleNamespace(
        **{name: name.lower() for name in parameter_names}
    )
    sdk.metanames = metanames
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai", sdk)
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai.metanames", metanames)

    credentials = Mock(return_value={"url": "https://watsonx.invalid"})
    constructor = Mock(return_value=None)
    monkeypatch.setattr(watsonx, "get_watsonx_credentials", credentials)
    monkeypatch.setattr(watsonx.WatsonxLLM, "__init__", constructor)
    return constructor, credentials


def _create_model(arg_string, additional_config):
    if additional_config is _OMITTED:
        return watsonx.WatsonxLLM.create_from_arg_string(arg_string)
    return watsonx.WatsonxLLM.create_from_arg_string(arg_string, additional_config)


@pytest.mark.parametrize("identifier", ["model_id", "deployment_id"])
@pytest.mark.parametrize("additional_config", _CONFIG_CASES)
def test_optional_configuration(identifier, additional_config, factory_dependencies):
    constructor, credentials = factory_dependencies

    model = _create_model(
        f"{identifier}=example-model,max_new_tokens=17", additional_config
    )

    assert isinstance(model, watsonx.WatsonxLLM)
    credentials.assert_called_once_with()
    constructor.assert_called_once_with(
        watsonx_credentials=credentials.return_value,
        model_id="example-model" if identifier == "model_id" else None,
        deployment_id="example-model" if identifier == "deployment_id" else None,
        generate_params={
            "decoding_method": "greedy",
            "max_new_tokens": 17,
            "return_options": {
                "generated_tokens": True,
                "input_tokens": True,
                "token_logprobs": True,
                "token_ranks": True,
            },
        },
    )


@pytest.mark.parametrize("identifier", ["model_id", "deployment_id"])
def test_configuration_overrides_parsed_arguments(identifier, factory_dependencies):
    constructor, _ = factory_dependencies
    additional_config = {
        identifier: "override-model",
        "temperature": 0.7,
        "max_new_tokens": 23,
    }
    original_config = additional_config.copy()

    _create_model(
        f"{identifier}=original-model,do_sample=True,temperature=0.2,max_new_tokens=17",
        additional_config,
    )

    constructor.assert_called_once()
    kwargs = constructor.call_args.kwargs
    assert kwargs[identifier] == "override-model"
    assert kwargs["generate_params"]["decoding_method"] == "sample"
    assert kwargs["generate_params"]["temperature"] == 0.7
    assert kwargs["generate_params"]["max_new_tokens"] == 23
    assert additional_config == original_config


@pytest.mark.parametrize("identifier", ["model_id", "deployment_id"])
def test_configuration_can_supply_identifier(identifier, factory_dependencies):
    constructor, _ = factory_dependencies

    _create_model("max_new_tokens=17", {identifier: "configured-model"})

    constructor.assert_called_once()
    assert constructor.call_args.kwargs[identifier] == "configured-model"


def test_explicit_none_values_keep_existing_behavior(factory_dependencies):
    constructor, _ = factory_dependencies
    additional_config = {"temperature": None, "max_new_tokens": None}
    original_config = additional_config.copy()

    _create_model(
        "model_id=example-model,do_sample=True,temperature=0.2,max_new_tokens=17",
        additional_config,
    )

    constructor.assert_called_once()
    params = constructor.call_args.kwargs["generate_params"]
    assert params["decoding_method"] == "sample"
    assert "temperature" not in params
    assert "max_new_tokens" not in params
    assert additional_config == original_config


@pytest.mark.parametrize("additional_config", _CONFIG_CASES)
def test_missing_identifier_raises_value_error(additional_config, factory_dependencies):
    constructor, credentials = factory_dependencies

    with pytest.raises(ValueError, match="'model_id' or 'deployment_id' is required"):
        _create_model("max_new_tokens=17", additional_config)

    credentials.assert_not_called()
    constructor.assert_not_called()
