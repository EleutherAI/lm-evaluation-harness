import base64
import io
import json
import pickle
import re
import zlib

from lm_eval.tasks.livecodebench.testing_util import run_test


FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

EVAL_TIMEOUT = 6


def doc_to_text(doc: dict) -> str:
    prompt = f"### Question:\n{doc['question_content']}\n\n"
    if doc["starter_code"]:
        prompt += (
            f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
            f"```python\n{doc['starter_code']}\n```\n\n"
        )
    else:
        prompt += (
            f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(text: str) -> str:
    matches = re.findall(r"```(?:\w+)?\n?(.*?)\n?```", text, re.DOTALL)
    return matches[0] if matches else ""


class _DataOnlyUnpickler(pickle.Unpickler):  # noqa: S301
    """Unpickler that rejects any GLOBAL/STACK_GLOBAL opcode.

    private_test_cases are zlib-compressed pickles; upstream decodes them
    with plain pickle.loads. The payload is only ever a JSON string, so
    refusing all class references keeps decoding equivalent for real data
    while preventing arbitrary code execution on a tampered blob.
    """

    def find_class(self, module, name):
        raise pickle.UnpicklingError("private_test_cases may only contain plain data")


def _decode_private_test_cases(doc: dict) -> list:
    blob = doc["private_test_cases"]
    if not blob:
        return []
    try:
        return json.loads(blob)
    except (json.JSONDecodeError, TypeError):
        raw = zlib.decompress(base64.b64decode(blob.encode("utf-8")))
        return json.loads(_DataOnlyUnpickler(io.BytesIO(raw)).load())


def _build_eval_sample(doc: dict) -> dict:
    tests = json.loads(doc["public_test_cases"]) + _decode_private_test_cases(doc)
    metadata = json.loads(doc["metadata"])
    return {
        "input_output": json.dumps(
            {
                "inputs": [t["input"] for t in tests],
                "outputs": [t["output"] for t in tests],
                "fn_name": metadata.get("func_name", None),
            }
        ),
    }


def process_results(doc: dict, results: list[list[str]]) -> dict:
    code = extract_code(results[0][0])
    if not code:
        return {"pass@1": 0.0}
    sample = _build_eval_sample(doc)
    res, _metadata = run_test(sample, test=code, timeout=EVAL_TIMEOUT)
    passed = bool(res) and all(r > 0 for r in res)
    return {"pass@1": float(passed)}
