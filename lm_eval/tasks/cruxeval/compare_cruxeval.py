"""
Comparison script: lm-evaluation-harness cruxeval vs. reference implementation.

This script takes pre-scored model generations in the reference cruxeval format
(https://github.com/facebookresearch/cruxeval) and re-scores them using the
lm-eval implementation, then reports any per-sample disagreements.

Reference generation format (JSON):
    {"sample_0": ["gen1", "gen2", ...], "sample_1": [...], ...}

Where each generation is already postprocessed by the reference pipeline:
  - output mode: a Python literal, e.g. "17" or '"hello"'
  - input mode:  a function call, e.g. 'f(16)' or 'f("ba", "nana")'

Usage:
    python scripts/compare_cruxeval.py \\
        --generations /path/to/cruxeval/samples/model_generations/sample_codellama-7b_temp0.2_output/generations.json \\
        --mode output \\
        --cruxeval_root /home/thomas/Work/cruxeval

    python scripts/compare_cruxeval.py \\
        --generations /path/to/cruxeval/samples/model_generations/sample_codellama-7b_temp0.2_input/generations.json \\
        --mode input \\
        --cruxeval_root /home/thomas/Work/cruxeval
"""

import argparse
import json
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import contextlib
import faulthandler
import io
import multiprocessing
import platform
import signal
import tempfile

import numpy as np


# ---------------------------------------------------------------------------
# Inlined check_correctness (mirrors lm_eval/tasks/cruxeval/utils.py and the
# reference cruxeval/evaluation/utils_execute.py — both are identical copies
# of OpenAI's human-eval execution harness).
# Inlining makes this script fully self-contained with no install required.
# ---------------------------------------------------------------------------

class _TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *a, **kw): raise OSError
    def readline(self, *a, **kw): raise OSError
    def readlines(self, *a, **kw): raise OSError
    def readable(self, *a, **kw): return False


class _redirect_stdin(contextlib._RedirectStream):
    _stream = "stdin"


@contextlib.contextmanager
def _chdir(root):
    if root == ".":
        yield; return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as d:
        with _chdir(d):
            yield d


@contextlib.contextmanager
def _time_limit(seconds):
    def handler(signum, frame):
        raise _TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


def _reliability_guard(maximum_memory_bytes=None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    faulthandler.disable()
    import builtins
    builtins.exit = None
    builtins.quit = None
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.kill = os.system = os.putenv = os.remove = os.removedirs = None
    os.rmdir = os.fchdir = os.setuid = os.fork = os.forkpty = None
    os.killpg = os.rename = os.renames = os.truncate = os.replace = None
    os.unlink = os.fchmod = os.fchown = os.chmod = os.chown = None
    os.chroot = os.lchflags = os.lchmod = os.lchown = os.getcwd = os.chdir = None
    import shutil
    shutil.rmtree = shutil.move = shutil.chown = None
    import subprocess
    subprocess.Popen = None
    if isinstance(__builtins__, dict):
        __builtins__["help"] = None
    else:
        __builtins__.help = None
    import sys
    sys.modules["ipdb"] = sys.modules["joblib"] = None
    sys.modules["resource"] = sys.modules["psutil"] = sys.modules["tkinter"] = None


def _unsafe_execute(check_program, result, timeout):
    with _create_tempdir():
        import os, shutil
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir_fn = os.chdir
        _reliability_guard()
        try:
            exec_globals = {}
            with _swallow_io():
                with _time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except _TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir_fn


def check_correctness(check_program, timeout=3):
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_unsafe_execute, args=(check_program, result, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result.append("timed out")
    return result[0] == "passed"


# ---------------------------------------------------------------------------
# Reference implementation (vendored to avoid sys.path hacks)
# We replicate the exact scoring logic from cruxeval/evaluation/utils_general.py
# so that we can compare without modifying the reference repo.
# ---------------------------------------------------------------------------

def _ref_pass_at_k(n, c, k):
    """Reference pass@k from cruxeval/evaluation/utils_general.py."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def _ref_evaluate_score(args):
    """
    Exact copy of evaluate_score from cruxeval/evaluation/utils_general.py.

    NOTE: invalid generations are *skipped* (not appended as False), so `n`
    in pass@k is the number of valid generations, not total generations.
    """
    gs, (c, i, o), mode = args
    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            pass  # skip — intentional reference behaviour
        elif mode == "output" and f"f({i})" in g:
            pass  # skip — intentional reference behaviour
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results


# ---------------------------------------------------------------------------
# lm-eval implementation (our scoring logic)
# ---------------------------------------------------------------------------

def _lmeval_evaluate_score_output(gs, doc):
    """
    Score output-prediction generations using our lm-eval assertion format:
        {code}
        assert f({input}) == {predicted_output}

    The reference generations for output mode are already the extracted literal
    value (e.g. '17', '"hello"'), so we use them directly as the predicted value.
    """
    results = []
    for g in gs:
        code_to_execute = f"{doc['code']}\nassert f({doc['input']}) == {g}"
        results.append(check_correctness(code_to_execute, 3))
    return results


def _lmeval_evaluate_score_input(gs, doc):
    """
    Score input-prediction generations using our lm-eval assertion format:
        {code}
        assert f({predicted_input}) == {expected_output}

    The reference generations for input mode are function calls like 'f(16)',
    so we strip the outer f(...) to get just the arguments.
    """
    results = []
    for g in gs:
        # Extract the arguments from 'f(args)' → 'args'
        if g.startswith("f(") and g.endswith(")"):
            args = g[2:-1]
        elif "f(" in g:
            try:
                start = g.index("f(") + 2
                paren = 1
                end = start
                while end < len(g) and paren > 0:
                    if g[end] == "(":
                        paren += 1
                    elif g[end] == ")":
                        paren -= 1
                    end += 1
                args = g[start:end - 1]
            except ValueError:
                args = g
        else:
            args = g

        code_to_execute = f"{doc['code']}\nassert f({args}) == {doc['output']}"
        results.append(check_correctness(code_to_execute, 3))
    return results


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def load_dataset(cruxeval_root: Path) -> list[dict]:
    jsonl_path = cruxeval_root / "data" / "cruxeval.jsonl"
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def compute_pass_at_k(all_scores: list[list[bool]], k: int) -> float:
    vals = []
    for scores in all_scores:
        n = len(scores)
        c = sum(scores)
        vals.append(_ref_pass_at_k(n, c, k))
    return float(np.mean(vals)) * 100


def compare(generations_path: str, mode: str, cruxeval_root: Path, limit: int | None):
    print(f"\n{'='*60}")
    print(f"  CRUXEval implementation comparison")
    print(f"  mode       : {mode}")
    print(f"  generations: {generations_path}")
    print(f"{'='*60}\n")

    generations: dict = json.load(open(generations_path))
    dataset = load_dataset(cruxeval_root)

    if limit:
        dataset = dataset[:limit]
        print(f"  [--limit {limit}] running on first {limit} samples only\n")

    references = [(doc["code"], doc["input"], doc["output"]) for doc in dataset]
    n_samples = len(dataset)

    try:
        gens_list = [generations[f"sample_{i}"] for i in range(n_samples)]
    except KeyError as e:
        sys.exit(f"ERROR: missing key {e} in generations file. "
                 "Expected keys sample_0 … sample_{n-1}.")

    # ---- Reference scoring ------------------------------------------------
    print("Scoring with reference implementation …")
    with ProcessPoolExecutor() as ex:
        ref_scores = list(ex.map(
            _ref_evaluate_score,
            zip(gens_list, references, [mode] * n_samples),
        ))

    # ---- lm-eval scoring --------------------------------------------------
    print("Scoring with lm-eval implementation …")
    lmeval_scores = []
    for gs, doc in zip(gens_list, dataset):
        if mode == "output":
            lmeval_scores.append(_lmeval_evaluate_score_output(gs, doc))
        else:
            lmeval_scores.append(_lmeval_evaluate_score_input(gs, doc))

    # ---- Per-sample comparison --------------------------------------------
    disagreements = []
    for i, (ref, lme) in enumerate(zip(ref_scores, lmeval_scores)):
        ref_any = any(ref)
        lme_any = any(lme)
        if ref_any != lme_any:
            disagreements.append({
                "sample": i,
                "ref_pass": ref_any,
                "lmeval_pass": lme_any,
                "ref_results": ref,
                "lmeval_results": lme,
                "code": dataset[i]["code"][:80] + "…",
                "input": dataset[i]["input"],
                "output": dataset[i]["output"],
                "generations": gens_list[i],
            })

    # ---- Aggregate pass@k -------------------------------------------------
    for k in [1, 5]:
        ref_p   = compute_pass_at_k(ref_scores,   k)
        lmeval_p = compute_pass_at_k(lmeval_scores, k)
        match = "✓" if abs(ref_p - lmeval_p) < 0.1 else "✗"
        print(f"  pass@{k}: reference={ref_p:.1f}%  lm-eval={lmeval_p:.1f}%  {match}")

    # ---- Disagreement report ----------------------------------------------
    print(f"\n  Disagreements (sample-level any-pass): {len(disagreements)} / {n_samples}")
    if disagreements:
        print()
        for d in disagreements[:10]:  # show at most 10
            print(f"  sample_{d['sample']}:")
            print(f"    code    : {d['code']}")
            print(f"    input   : {d['input']}")
            print(f"    output  : {d['output']}")
            print(f"    gens    : {d['generations']}")
            print(f"    ref     : {d['ref_results']} (any={d['ref_pass']})")
            print(f"    lm-eval : {d['lmeval_results']} (any={d['lmeval_pass']})")
            print()
        if len(disagreements) > 10:
            print(f"  … and {len(disagreements) - 10} more.")

    return len(disagreements) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--generations", required=True,
        help="Path to a reference-format generations JSON file.",
    )
    parser.add_argument(
        "--mode", choices=["input", "output"],
        help="Evaluation mode. Inferred from filename if omitted.",
    )
    parser.add_argument(
        "--cruxeval_root", default=str(Path(__file__).parent.parent.parent / "cruxeval"),
        help="Path to the root of the reference cruxeval repo. "
             "Default: ../cruxeval relative to lm-evaluation-harness.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N samples (useful for quick checks).",
    )
    args = parser.parse_args()

    # Infer mode from filename if not given
    if args.mode is None:
        path_lower = args.generations.lower()
        if "input" in path_lower:
            args.mode = "input"
        elif "output" in path_lower:
            args.mode = "output"
        else:
            sys.exit("ERROR: --mode is required when it cannot be inferred from the filename.")

    cruxeval_root = Path(args.cruxeval_root)
    if not (cruxeval_root / "data" / "cruxeval.jsonl").exists():
        sys.exit(f"ERROR: cruxeval dataset not found at {cruxeval_root}/data/cruxeval.jsonl")

    ok = compare(args.generations, args.mode, cruxeval_root, args.limit)
    sys.exit(0 if ok else 1)
