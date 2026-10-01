import sys

import pytest

from lm_eval.models.mamba_lm import MambaLMWrapper


@pytest.mark.parametrize("method_name", ["_get_config", "_create_model"])
def test_missing_dependency_recommends_available_install(method_name, monkeypatch):
    for module_name in (
        "mamba_ssm",
        "mamba_ssm.models",
        "mamba_ssm.models.mixer_seq_simple",
        "mamba_ssm.utils",
        "mamba_ssm.utils.hf",
    ):
        monkeypatch.setitem(sys.modules, module_name, None)

    model = object.__new__(MambaLMWrapper)
    model.is_hf = False

    with pytest.raises(ModuleNotFoundError) as exc_info:
        getattr(model, method_name)("state-spaces/mamba-130m")

    message = str(exc_info.value)
    assert "pip install mamba-ssm --no-build-isolation" in message
    assert "lm-eval[mamba]" not in message
    assert ".[mamba]" not in message
