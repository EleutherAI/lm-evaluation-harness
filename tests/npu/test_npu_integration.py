"""Pytest suite for the Rebellions NPU integration with lm_eval.

These tests use ``assert`` (not ``return``) so that failures surface correctly
under pytest. Checks split into two groups:

* Pure-Python contracts (imports, registry wiring, npu_utils return types) run
  on any host and must genuinely pass/fail.
* Hardware / SDK dependent checks (the ``optimum-rbln`` SDK, the ``rbln-stat``
  CLI) are *skipped* when unavailable rather than silently reported as passing.
"""

import shutil

import pytest


def test_rbln_models_importable():
    """RBLN model classes import and sit in the expected class hierarchy."""
    from lm_eval.api.model import TemplateLM
    from lm_eval.models.optimum_rbln import RBLNLM
    from lm_eval.models.optimum_rbln_vlm import RBLNVLM

    assert issubclass(RBLNLM, TemplateLM)
    assert issubclass(RBLNVLM, RBLNLM)


def test_rbln_models_registered():
    """``rbln`` and ``rbln-vlm`` resolve to the expected classes in the registry."""
    from lm_eval.api.registry import get_model
    from lm_eval.models.optimum_rbln import RBLNLM
    from lm_eval.models.optimum_rbln_vlm import RBLNVLM

    assert get_model("rbln") is RBLNLM
    assert get_model("rbln-vlm") is RBLNVLM


def test_npu_utils_contract():
    """Detection helpers honour their return types and stay mutually consistent.

    These must degrade gracefully on a host with no NPU (0 / False / []) instead
    of raising, so they run everywhere.
    """
    from lm_eval.npu_utils import (
        detect_npu_devices,
        get_npu_count,
        is_npu_available,
    )

    count = get_npu_count()
    available = is_npu_available()
    devices = detect_npu_devices()

    assert isinstance(count, int) and count >= 0
    assert isinstance(available, bool)
    assert isinstance(devices, list)
    # The two derived views must agree with the device list.
    assert available == (count > 0)
    assert len(devices) == count


def test_rbln_sdk_available():
    """The RBLN SDK (optimum-rbln) is importable; skipped when not installed."""
    pytest.importorskip(
        "optimum.rbln",
        reason="optimum-rbln not installed (no RBLN SDK on this host)",
    )


@pytest.mark.skipif(
    shutil.which("rbln-stat") is None,
    reason="rbln-stat not on PATH (not an RBLN NPU host)",
)
def test_rbln_stat_runs():
    """On an NPU host, rbln-stat runs and feeds the detection helpers."""
    from lm_eval.npu_utils import get_npu_count, is_npu_available, run_rbln_stat

    stat = run_rbln_stat()
    assert stat is not None, "rbln-stat is on PATH but returned no parsable output"
    assert is_npu_available() == (get_npu_count() > 0)
