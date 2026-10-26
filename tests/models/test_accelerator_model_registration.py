import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize(
    "alias,module_name,class_name",
    [
        ("ipex", "lm_eval.models.optimum_ipex", "IPEXLM"),
        ("neuronx", "lm_eval.models.neuron_optimum", "NEURON_HF"),
    ],
)
@pytest.mark.parametrize("import_first", [False, True])
def test_accelerator_adapter_registration(alias, module_name, class_name, import_first):
    # Each route must work before another test imports/registers the adapter.
    code = textwrap.dedent(
        f"""
        import importlib
        import sys
        from lm_eval.api.model import LM
        from lm_eval.api.registry import get_model
        import lm_eval.models

        assert {module_name!r} not in sys.modules
        if {import_first!r}:
            importlib.import_module({module_name!r})
        adapter = get_model({alias!r})
        expected = getattr(importlib.import_module({module_name!r}), {class_name!r})
        assert adapter is expected
        assert issubclass(adapter, LM)
        assert get_model({alias!r}) is adapter
        """
    )
    result = subprocess.run(  # noqa: S603 - fixed test code in the current interpreter
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
