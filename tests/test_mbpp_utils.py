import os

import pytest


# `mbpp.utils` loads the `code_eval` metric at import time, which requires the
# `evaluate` package and HF_ALLOW_CODE_EVAL=1.
os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
mbpp_utils = pytest.importorskip(
    "lm_eval.tasks.mbpp.utils",
    reason="mbpp needs the `evaluate` package and the `code_eval` metric",
)

extract_code_blocks = mbpp_utils.extract_code_blocks


@pytest.mark.parametrize(
    "response, expected",
    [
        # Continuation of the "```python\n" generation prefix, closed by the model.
        pytest.param(
            "def add(a, b):\n    return a + b\n```",
            "def add(a, b):\n    return a + b",
            id="prefix-continuation",
        ),
        # Model restates the language tag before the code.
        pytest.param(
            "python\ndef add(a, b):\n    return a + b\n```",
            "def add(a, b):\n    return a + b",
            id="language-tag",
        ),
        # Unfenced code starting with a keyword: before the fix "def" / "from" /
        # "import" was swallowed as the language tag and the code became invalid.
        pytest.param(
            "def add(a, b):\n    return a + b\n```\nExplanation.",
            "def add(a, b):\n    return a + b",
            id="def-not-swallowed",
        ),
        pytest.param(
            "from math import gcd\ndef f(a, b):\n    return gcd(a, b)\n```",
            "from math import gcd\ndef f(a, b):\n    return gcd(a, b)",
            id="from-not-swallowed",
        ),
        pytest.param(
            "import re\ndef f(s):\n    return re.sub('x', '', s)\n```",
            "import re\ndef f(s):\n    return re.sub('x', '', s)",
            id="import-not-swallowed",
        ),
    ],
)
def test_extract_code_blocks(response, expected):
    assert extract_code_blocks(response) == expected
