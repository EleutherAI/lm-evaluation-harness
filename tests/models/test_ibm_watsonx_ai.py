"""Watsonx backend tests that need neither the optional SDK nor credentials."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from lm_eval.api.registry import get_model
from lm_eval.models.ibm_watsonx_ai import (
    WatsonxLLM,
    _verify_credentials,
    get_watsonx_credentials,
)


@pytest.fixture
def watsonx_env(monkeypatch):
    """Isolate credentials and prevent dotenv from reading any real .env file."""
    for key in (
        "USERNAME",
        "PASSWORD",
        "API_KEY",
        "TOKEN",
        "URL",
        "PROJECT_ID",
        "SPACE_ID",
    ):
        monkeypatch.delenv(f"WATSONX_{key}", raising=False)
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    monkeypatch.setenv("WATSONX_API_KEY", "test-key")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "test-project")
    monkeypatch.setenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
    get_watsonx_credentials.cache_clear()
    yield dotenv.load_dotenv
    get_watsonx_credentials.cache_clear()


@pytest.mark.parametrize("url", [None, ""])
def test_missing_url_has_actionable_validation_error(watsonx_env, monkeypatch, url):
    if url is None:
        monkeypatch.delenv("WATSONX_URL")
    else:
        monkeypatch.setenv("WATSONX_URL", url)

    with pytest.raises(ValueError, match=r"url \(WATSONX_URL\)"):
        get_watsonx_credentials()


@pytest.mark.parametrize(
    ("url", "expected_instance_id"),
    [
        ("https://us-south.ml.cloud.ibm.com", None),
        ("https://eu-de.ml.cloud.ibm.com", None),
        ("https://watsonx.example.com", "openshift"),
    ],
)
def test_credentials_select_cloud_or_openshift(
    watsonx_env, monkeypatch, url, expected_instance_id
):
    monkeypatch.setenv("WATSONX_URL", url)

    creds = get_watsonx_credentials()

    assert creds["url"] == url
    assert creds["apikey"] == "test-key"
    assert creds["project_id"] == "test-project"
    assert creds.get("instance_id") == expected_instance_id
    watsonx_env.assert_called_once_with()


@pytest.mark.parametrize("scope", ["project_id", "space_id"])
@pytest.mark.parametrize(
    "authentication",
    [{"apikey": "test-key"}, {"username": "test-user", "password": "test-password"}],
)
def test_verify_credentials_accepts_supported_auth_and_scope(authentication, scope):
    _verify_credentials(
        {"url": "https://watsonx.example.com", scope: "test-scope", **authentication}
    )


@pytest.mark.parametrize(
    "authentication", [{}, {"username": "test-user"}, {"password": "test-password"}]
)
def test_verify_credentials_rejects_incomplete_authentication(authentication):
    with pytest.raises(ValueError, match="WATSONX_API_KEY"):
        _verify_credentials(
            {
                "url": "https://watsonx.example.com",
                "project_id": "test-project",
                **authentication,
            }
        )


def test_verify_credentials_reports_all_missing_fields():
    with pytest.raises(ValueError) as exc_info:
        _verify_credentials({})

    for key in (
        "WATSONX_API_KEY",
        "WATSONX_URL",
        "WATSONX_PROJECT_ID",
        "WATSONX_SPACE_ID",
    ):
        assert key in str(exc_info.value)


def test_env_credentials_support_username_password_and_space(watsonx_env, monkeypatch):
    monkeypatch.delenv("WATSONX_API_KEY")
    monkeypatch.delenv("WATSONX_PROJECT_ID")
    monkeypatch.setenv("WATSONX_USERNAME", "test-user")
    monkeypatch.setenv("WATSONX_PASSWORD", "test-password")
    monkeypatch.setenv("WATSONX_SPACE_ID", "test-space")

    creds = get_watsonx_credentials()

    assert creds["username"] == "test-user"
    assert creds["password"] == "test-password"  # noqa: S105 - fake test credential
    assert creds["space_id"] == "test-space"
    assert creds["apikey"] is None
    assert creds["project_id"] is None


def test_env_credentials_warn_about_conflicting_auth(watsonx_env, monkeypatch):
    monkeypatch.setenv("WATSONX_USERNAME", "test-user")
    monkeypatch.setenv("WATSONX_PASSWORD", "test-password")

    with pytest.warns(UserWarning, match="username.*,.*password.*,.*apikey"):
        creds = get_watsonx_credentials()

    assert creds["apikey"] == "test-key"


def test_missing_url_and_auth_report_both_problems(watsonx_env, monkeypatch):
    monkeypatch.delenv("WATSONX_URL")
    monkeypatch.delenv("WATSONX_API_KEY")

    with pytest.raises(ValueError) as exc_info:
        get_watsonx_credentials()

    assert "WATSONX_URL" in str(exc_info.value)
    assert "WATSONX_API_KEY" in str(exc_info.value)


@pytest.fixture
def watsonx_sdk(monkeypatch):
    """Mock the SDK boundary so backend initialization stays fully offline."""
    sdk = ModuleType("ibm_watsonx_ai")
    sdk.Credentials = Mock()
    sdk.APIClient = Mock()
    foundation_models = ModuleType("ibm_watsonx_ai.foundation_models")
    foundation_models.ModelInference = Mock()
    monkeypatch.setitem(sys.modules, "ibm_watsonx_ai", sdk)
    monkeypatch.setitem(
        sys.modules, "ibm_watsonx_ai.foundation_models", foundation_models
    )
    return sdk, foundation_models


@pytest.mark.parametrize(
    "identifier", [{"model_id": "test-model"}, {"deployment_id": "test-deployment"}]
)
def test_backend_initialization_passes_credentials_and_identifier(
    watsonx_sdk, identifier
):
    sdk, foundation_models = watsonx_sdk
    creds = {
        "apikey": "test-key",
        "url": "https://watsonx.example.com",
        "project_id": "test-project",
    }
    params = {"max_new_tokens": 12}

    model = WatsonxLLM(watsonx_credentials=creds, generate_params=params, **identifier)

    sdk.Credentials.from_dict.assert_called_once_with(creds)
    sdk.APIClient.assert_called_once_with(
        credentials=sdk.Credentials.from_dict.return_value, project_id="test-project"
    )
    foundation_models.ModelInference.assert_called_once_with(
        model_id=identifier.get("model_id"),
        deployment_id=identifier.get("deployment_id"),
        api_client=sdk.APIClient.return_value,
    )
    assert model.model is foundation_models.ModelInference.return_value
    assert model.generate_params == params
    assert model.rank == 0
    assert model.world_size == 1


def test_backend_uses_validated_environment_credentials(watsonx_env, watsonx_sdk):
    sdk, _ = watsonx_sdk

    WatsonxLLM(model_id="test-model")

    passed_creds = sdk.Credentials.from_dict.call_args.args[0]
    assert passed_creds["url"] == "https://us-south.ml.cloud.ibm.com"
    assert passed_creds["apikey"] == "test-key"
    watsonx_env.assert_called_once_with()


def test_backend_missing_url_fails_before_sdk_client_creation(
    watsonx_env, watsonx_sdk, monkeypatch
):
    sdk, foundation_models = watsonx_sdk
    monkeypatch.delenv("WATSONX_URL")

    with pytest.raises(ValueError, match="WATSONX_URL"):
        WatsonxLLM(model_id="test-model")

    sdk.Credentials.from_dict.assert_not_called()
    sdk.APIClient.assert_not_called()
    foundation_models.ModelInference.assert_not_called()


def test_watsonx_registry_resolves_backend():
    assert get_model("watsonx_llm") is WatsonxLLM
