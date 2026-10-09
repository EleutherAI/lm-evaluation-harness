import base64
import json
import os
import pickle
import subprocess
import sys
import zlib

import pytest


utils = pytest.importorskip(
    "lm_eval.tasks.livecodebench.utils",
    reason="livecodebench utils import the vendored checker",
)


STDIN_DOC = {
    "question_content": "Read an integer n and print n squared.",
    "starter_code": "",
    "public_test_cases": json.dumps(
        [{"input": "3\n", "output": "9\n", "testtype": "stdin"}]
    ),
    "private_test_cases": base64.b64encode(
        zlib.compress(
            pickle.dumps(
                json.dumps([{"input": "5\n", "output": "25\n", "testtype": "stdin"}])
            )
        )
    ).decode(),
    "metadata": "{}",
}

CALL_DOC = {
    "question_content": "Return the sum of a list.",
    "starter_code": "class Solution:\n    def total(self, xs: List[int]) -> int:\n        ",
    "public_test_cases": json.dumps(
        [{"input": "[1, 2, 3]", "output": "6", "testtype": "functional"}]
    ),
    "private_test_cases": base64.b64encode(
        zlib.compress(
            pickle.dumps(
                json.dumps(
                    [{"input": "[10, -4]", "output": "6", "testtype": "functional"}]
                )
            )
        )
    ).decode(),
    "metadata": json.dumps({"func_name": "total"}),
}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("```python\nprint(1)\n```", "print(1)"),
        ("prefix\n```\nx = 2\n```\nsuffix", "x = 2"),
        ("```python\na=1\n```\n```python\nb=2\n```", "a=1"),
        ("no code fences here", ""),
        ("", ""),
    ],
)
def test_extract_code(text, expected):
    assert utils.extract_code(text) == expected


def test_doc_to_text_stdin_variant():
    prompt = utils.doc_to_text(STDIN_DOC)
    assert prompt.startswith("### Question:\nRead an integer")
    assert "Read the inputs from stdin" in prompt
    assert "starter code" not in prompt


def test_doc_to_text_call_variant_includes_starter():
    prompt = utils.doc_to_text(CALL_DOC)
    assert "starter code" in prompt
    assert "```python\nclass Solution:" in prompt


def test_decode_private_test_cases_accepts_json_and_pickle():
    doc = dict(STDIN_DOC)
    doc["private_test_cases"] = json.dumps([{"input": "7", "output": "49"}])
    assert utils._decode_private_test_cases(doc) == [{"input": "7", "output": "49"}]
    decoded = utils._decode_private_test_cases(STDIN_DOC)
    assert decoded[0]["output"] == "25\n"


def test_decode_private_test_cases_rejects_class_payload():
    payload = b"cposix\nsystem\n(S'echo PWNED'\ntR."
    doc = {"private_test_cases": base64.b64encode(zlib.compress(payload)).decode()}
    with pytest.raises(pickle.UnpicklingError):
        utils._decode_private_test_cases(doc)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _score_in_subprocess(doc, generation):
    """process_results runs the checker's reliability_guard, which disables
    os functions in-process; keep it out of the pytest process.
    """
    script = (
        "import sys\n"
        f"sys.path.insert(0, {json.dumps(REPO_ROOT)})\n"
        "from lm_eval.tasks.livecodebench.utils import process_results\n"
        f"import json; doc = json.loads({json.dumps(json.dumps(doc))})\n"
        f"print(process_results(doc, [{json.dumps(generation)}]))\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().replace("'", '"'))


def test_process_results_scores_stdin():
    good = "```python\nn = int(input())\nprint(n * n)\n```"
    bad = "```python\nprint(0)\n```"
    assert _score_in_subprocess(STDIN_DOC, good) == {"pass@1": 1.0}
    assert _score_in_subprocess(STDIN_DOC, bad) == {"pass@1": 0.0}


def test_process_results_scores_call_based():
    good = "```python\nclass Solution:\n    def total(self, xs):\n        return sum(xs)\n```"
    assert _score_in_subprocess(CALL_DOC, good) == {"pass@1": 1.0}


def test_process_results_empty_generation_fails():
    assert _score_in_subprocess(STDIN_DOC, "no code here") == {"pass@1": 0.0}
