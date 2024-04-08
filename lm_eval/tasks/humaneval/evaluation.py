import os
from random import random
import subprocess

def code_judgement(doc, prediction):
    """
    Return True if the code runs without error, else return False.
    doc: sample
    prediction: str, code completion
    """
    completed_code = doc["prompt"] + prediction + "\n" + doc["test"]
    entry_point = doc["entry_point"]
    completed_code += f"\n\ncheck({entry_point})\n"
    i = random()
    try:
        os.makedirs("tmp_save", exist_ok=True)
        name = f"{completed_code[:20]}_{i}"
        with open(f"tmp_save/{name}.py", "w") as f:
            f.write(completed_code)
        # run "python tmp_save.py" command
        # save the exit result
        result = subprocess.run(["python", f"tmp_save/{name}.py"], capture_output=True, text=True, timeout=3.)
        if result.returncode != 0:
            raise Exception(result.stderr)
    except Exception as e:
        if str(e).rstrip().rsplit("\n")[-1].startswith("AssertionError"):
            return False

        print(f"[{i}] Error occurred while running the code:\n")
        print(f"[{i}]", completed_code)
        print(f"[{i}] {e}\n")
        return False
    return True


def process_results(doc, results):
    # if not doc["prompt"].strip().startswith("def tri(n):"):
    #     return {"pass@1": 1}
    prediction = results[0]
    # deal with indentation

    prediction = "    " + prediction.lstrip()
    return {"pass@1": code_judgement(doc, prediction)}


if __name__ == '__main__':  # testing
    doc = {
        "prompt": 'def add(a, b):\n    """\n    Given two numbers a and b, return their sum.\n    """\n',
        "test": "\n\ndef check(fn):\n    assert fn(1, 2) == 3\n    assert fn(2, 3) == 5\n    assert fn(3, 4) == 7\n    "
                "assert fn(4, 5) == 9\n    assert fn(5, 6) == 11\n    assert fn(6, 7) == 13\n",
        "entry_point": "add"
    }
    results = ["    return a + b\n"]
    print(process_results(doc, results))
