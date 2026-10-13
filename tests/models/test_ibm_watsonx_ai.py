import sys
from types import SimpleNamespace

import pytest

from lm_eval.models.ibm_watsonx_ai import (
    _verify_credentials,
    get_watsonx_credentials,
)


@pytest.fixture(autouse=True)
def clear_cached_credentials():
    get_watsonx_credentials.cache_clear()
    yield
    get_watsonx_credentials.cache_clear()


@pytest.mark.parametrize(
    "credentials",
    [
        {"apikey": "secret", "url": "https://example.com", "project_id": "p"},
        {
            "username": "user",
            "password": "password",
            "url": "https://example.com",
            "space_id": "s",
        },
    ],
)
def test_verify_credentials_accepts_supported_authentication(credentials):
    _verify_credentials(credentials)


@pytest.mark.parametrize(
    ("credentials", "missing"),
    [
        ({"url": "https://example.com", "project_id": "p"}, "apikey"),
        ({"apikey": "secret", "project_id": "p"}, "url"),
        ({"apikey": "secret", "url": "https://example.com"}, "project_id"),
    ],
)
def test_verify_credentials_reports_missing_requirements(credentials, missing):
    with pytest.raises(ValueError, match=missing):
        _verify_credentials(credentials)


def test_get_watsonx_credentials_reports_missing_url(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=lambda: None)
    )
    for name in (
        "WATSONX_USERNAME",
        "WATSONX_PASSWORD",
        "WATSONX_TOKEN",
        "WATSONX_URL",
        "WATSONX_SPACE_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("WATSONX_API_KEY", "secret")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "project")

    with pytest.raises(ValueError, match=r"url \(WATSONX_URL\)"):
        get_watsonx_credentials()
